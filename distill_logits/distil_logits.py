import os
import logging
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import yaml
from datetime import datetime

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


class PeriodicTestCallback(TrainerCallback):
    """Callback to run periodic test evaluation during training."""

    def __init__(self, test_dataset, tokenizer, eval_steps=500, num_test_samples=5):
        """Initialize the callback.
        
        Args:
            test_dataset: The test dataset to evaluate on
            tokenizer: The tokenizer to use for generation
            eval_steps: Number of steps between evaluations
            num_test_samples: Number of samples to test
        """
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_test_samples = num_test_samples
        self.test_results = []

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs.get('model')
            if model is None:
                return

            logger.info(f"\n{'='*70}")
            logger.info(f"PERIODIC TEST EVALUATION - Step {state.global_step}")
            logger.info(f"{'='*70}")

            try:
                model.eval()
                with torch.no_grad():
                    # Sample test examples from the test dataset
                    import random
                    test_indices = random.sample(range(len(self.test_dataset)),
                                                 min(self.num_test_samples, len(self.test_dataset)))

                    total_loss = 0
                    for idx, sample_idx in enumerate(test_indices, 1):
                        sample = self.test_dataset[sample_idx]

                        # Prepare input
                        input_ids = torch.tensor([sample['input_ids']]).to(model.device)
                        attention_mask = torch.tensor([sample['attention_mask']]).to(model.device)

                        # Forward pass
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = outputs.loss if hasattr(outputs, 'loss') else None

                        if loss is not None:
                            total_loss += loss.item()
                            logger.info(
                                f"  Sample {idx}/{self.num_test_samples}: Loss = {loss.item():.4f}")

                    avg_loss = total_loss / self.num_test_samples if self.num_test_samples > 0 else 0
                    logger.info(f"\nAverage Test Loss: {avg_loss:.4f}")

                    # Store result
                    self.test_results.append({
                        'step': state.global_step,
                        'avg_loss': avg_loss,
                        'timestamp': datetime.now().isoformat()
                    })

                model.train()
            except Exception as e:
                logger.error(f"Error during periodic test evaluation: {e}", exc_info=True)
                model.train()

            logger.info(f"{'='*70}\n")


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

    # Add periodic test evaluation callback
    eval_steps = config["training"].get("eval_steps", 500)
    test_callback = PeriodicTestCallback(test_dataset=tokenized_dataset["test"],
                                         tokenizer=student_tokenizer,
                                         eval_steps=eval_steps,
                                         num_test_samples=5)
    trainer.add_callback(test_callback)
    logger.info(f"Added periodic test callback (every {eval_steps} steps)")

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model(config["training"]["output_dir"])
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
