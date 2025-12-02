import os
import logging
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator

from config import CONFIG
from data_processing import load_and_preprocess_dataset, process_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiLayerAdaptationLayer(torch.nn.Module):
    """Multi-layer adaptation layer for hidden state distillation."""

    def __init__(self,
                 student_dim,
                 teacher_dim,
                 num_student_layers,
                 num_teacher_layers,
                 dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states


class CustomSFTTrainer(SFTTrainer):
    """Custom trainer for hidden state-based knowledge distillation."""

    def __init__(self, *args, **kwargs):
        self.remove_unused_columns = kwargs.pop('remove_unused_columns', None)
        self.max_seq_length = kwargs.get('max_seq_length', 1024)
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        labels = inputs["labels"]

        student_outputs = model(**student_inputs, labels=labels, output_hidden_states=True)

        original_loss = student_outputs.loss

        self.teacher_model = self.teacher_model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model,
                                                             'module') else self.teacher_model

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["teacher_input_ids"],
                "attention_mask": inputs["teacher_attention_mask"],
            }

            teacher_outputs = teacher_model(**teacher_inputs, output_hidden_states=True)

        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs,
                                             original_loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]

            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}"
                )

            student_probs = F.softmax(adapted_student_hidden_states[student_hidden] /
                                      self.config_dict["distillation"]["temperature"],
                                      dim=-1)
            teacher_probs = F.softmax(teacher_hidden /
                                      self.config_dict["distillation"]["temperature"],
                                      dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] /
                              self.config_dict["distillation"]["temperature"],
                              dim=-1),
                teacher_probs,
                reduction='batchmean') * (self.config_dict["distillation"]["temperature"]**2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = self.config_dict["distillation"]["alpha"] * scaled_loss_kd + (
            1 - self.config_dict["distillation"]["alpha"]) * original_loss
        return total_loss


def main():
    """Main training function."""
    logger.info("Starting distill_hidden training...")

    # Use configuration
    config = CONFIG

    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info(f"Project name: {config['project_name']}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Load and preprocess dataset
    logger.info("Loading and preprocessing dataset...")
    dataset = load_and_preprocess_dataset(config)
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Load tokenizers
    logger.info(
        f"Loading tokenizers: teacher={config['models']['teacher']}, student={config['models']['student']}"
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

    # Apply chat template to student tokenizer
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Process and tokenize the dataset
    logger.info("Processing and tokenizing dataset...")
    dataset = process_dataset(dataset, student_tokenizer, teacher_tokenizer, config)
    logger.info(f"Dataset processed with {len(dataset)} samples")

    # Load models with configurable flash attention
    logger.info("Loading models...")
    model_kwargs = {
        "torch_dtype":
        torch.bfloat16 if config["training"]["bf16"] else
        (torch.float16 if config["training"]["fp16"] else torch.float32)
    }
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash attention 2")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"],
                                                         **model_kwargs).to(device)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"],
                                                         **model_kwargs).to(device)
    logger.info("Models loaded successfully")

    # Create adaptation layer
    logger.info("Creating multi-layer adaptation layer...")
    adaptation_layer = MultiLayerAdaptationLayer(student_model.config.hidden_size,
                                                 teacher_model.config.hidden_size,
                                                 student_model.config.num_hidden_layers,
                                                 teacher_model.config.num_hidden_layers,
                                                 dtype=torch.bfloat16).to(device)
    logger.info(f"Adaptation layer created with {len(adaptation_layer.projections)} projections")

    # Training arguments
    logger.info("Setting up training arguments...")
    training_arguments = TrainingArguments(
        **config["training"],
        remove_unused_columns=False,
    )

    # Create the custom SFT Trainer
    logger.info("Creating trainer...")
    trainer = CustomSFTTrainer(
        model=student_model,
        train_dataset=dataset,
        max_seq_length=config["tokenizer"]["max_length"],
        tokenizer=student_tokenizer,
        args=training_arguments,
        packing=config["training"].get("packing", False),
    )

    # Store config in trainer for access in loss computation
    trainer.config_dict = config

    # Add these attributes to the trainer
    trainer.teacher_model = teacher_model
    trainer.adaptation_layer = adaptation_layer
    trainer.student_tokenizer = student_tokenizer
    trainer.teacher_tokenizer = teacher_tokenizer

    # Prepare for distributed training
    logger.info("Preparing for distributed training...")
    trainer = accelerator.prepare(trainer)

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model(config["training"]["output_dir"])

    # Save the adaptation layer
    adaptation_layer_path = os.path.join(config["training"]["output_dir"], "adaptation_layer.pth")
    logger.info(f"Saving adaptation layer to {adaptation_layer_path}")
    torch.save(adaptation_layer.state_dict(), adaptation_layer_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
