"""Data processing utilities for distill_hidden training."""

import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_and_preprocess_dataset(config):
    """Load and preprocess dataset from configuration."""
    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    if config["dataset"].get("num_samples"):
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    return dataset


def prepare_dataset(example, student_tokenizer, teacher_tokenizer, config):
    """Prepare dataset with both student and teacher tokenized inputs."""
    system = "You are a helpful assistant chatbot."
    conversations = example['conversations']

    message = [{"role": "system", "content": system}]

    for conversation in conversations:
        if conversation.get('from') == 'human':
            message.append({"role": "user", "content": conversation.get('value', '')})
        elif conversation.get('from') == 'gpt':
            message.append({"role": "assistant", "content": conversation.get('value', '')})

    student_text = student_tokenizer.apply_chat_template(message,
                                                         tokenize=False,
                                                         add_generation_prompt=True)
    teacher_text = teacher_tokenizer.apply_chat_template(message,
                                                         tokenize=False,
                                                         add_generation_prompt=True)

    student_encodings = student_tokenizer(student_text,
                                          truncation=True,
                                          max_length=config["tokenizer"]["max_length"],
                                          padding='max_length')
    teacher_encodings = teacher_tokenizer(teacher_text,
                                          truncation=True,
                                          max_length=config["tokenizer"]["max_length"],
                                          padding='max_length')

    return {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "teacher_input_ids": teacher_encodings["input_ids"],
        "teacher_attention_mask": teacher_encodings["attention_mask"],
    }


def process_dataset(dataset, student_tokenizer, teacher_tokenizer, config):
    """Process dataset by formatting and tokenizing."""
    logger.info("Processing dataset with both student and teacher tokenizers...")
    original_columns = dataset.column_names

    dataset = dataset.map(
        lambda x: prepare_dataset(x, student_tokenizer, teacher_tokenizer, config),
        remove_columns=original_columns)
    logger.info(f"Dataset processing complete with {len(dataset)} samples")

    return dataset
