import os
import logging
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import yaml

from config import CONFIG
from data_processing import load_and_preprocess_dataset, prepare_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def freeze_student_spectrum(model, unfrozen_layers_file, logger):
    """Freeze student model layers based on spectrum configuration."""
    with open(unfrozen_layers_file, 'r') as file:
        unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']

    for name, param in model.named_parameters():
        if not any(layer in name for layer in unfrozen_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True

    logger.info(f"Froze layers based on spectrum configuration: {unfrozen_layers_file}")


def pad_logits(student_logits, teacher_logits):
    """Pad logits to match dimensions."""
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size),
                                 dtype=teacher_logits.dtype,
                                 device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1),
                teacher_logits) if student_size < teacher_size else (
                    student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


class LogitsTrainer(SFTTrainer):
    """Custom trainer for logits-based knowledge distillation."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)

        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model,
                                                             'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(model, student_outputs.logits, teacher_outputs.logits,
                                             inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device),
                                                    teacher_logits.to(device))

        student_logits_scaled = student_logits / self.config_dict["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / self.config_dict["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean') * (self.config_dict["distillation"]["temperature"]**
                                      2) / self.config_dict["tokenizer"]["max_length"]

        return self.config_dict["distillation"]["alpha"] * loss_kd + (
            1 - self.config_dict["distillation"]["alpha"]) * original_loss


def main():
    """Main training function."""
    logger.info("Starting distill_logits training...")

    # Use configuration
    config = CONFIG

    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    logger.info(f"Project name: {config['project_name']}")

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

    # Prepare and tokenize the dataset
    logger.info("Preparing and tokenizing dataset...")
    tokenized_dataset = prepare_dataset(dataset, student_tokenizer, config)
    logger.info(
        f"Tokenized dataset: train={len(tokenized_dataset['train'])}, test={len(tokenized_dataset['test'])}"
    )

    # Load models with configurable flash attention
    logger.info("Loading models...")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash attention 2")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"],
                                                         **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"],
                                                         **model_kwargs)
    logger.info("Models loaded successfully")

    # Optionally freeze layers of the student model based on spectrum configuration
    if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
        freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"], logger)
    else:
        logger.info(
            "Spectrum configuration not found. All layers of the student model will be trainable.")

    # Training arguments
    logger.info("Setting up training arguments...")
    training_arguments = TrainingArguments(**config["training"])

    # Create the custom SFT Trainer
    logger.info("Creating trainer...")
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
    )

    # Store config in trainer for access in loss computation
    trainer.config_dict = config

    # Add the teacher model to the trainer
    trainer.teacher_model = teacher_model

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model(config["training"]["output_dir"])
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
