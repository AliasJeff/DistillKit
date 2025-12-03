"""Data processing utilities for distill_logits training."""

import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_and_preprocess_dataset(config):
    """Load and preprocess dataset from configuration."""
    dataset = load_dataset(config["dataset"]["name"],
                           config["dataset"]["lang"],
                           split=config["dataset"]["split"])
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    return dataset


def sharegpt_format(example, student_tokenizer, config):
    """Convert ShareGPT format to chat template format."""
    conversations = example['conversations']
    message = []

    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'human':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'system':
                    message.insert(0, {"role": "system", "content": conversation.get('value', '')})

    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    text = student_tokenizer.apply_chat_template(message,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
    return {"text": text}


def freedom_intelligence_format(example, student_tokenizer, config):
    """Convert FreedomIntelligence format to chat template format."""
    # Question, Complex_CoT, Response
    message = [{
        "role": "user",
        "content": example['Question']
    }, {
        "role": "assistant",
        "content": example['Response']
    }]

    text = student_tokenizer.apply_chat_template(message,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
    return {"text": text}


def tokenize_function(examples, student_tokenizer, config):
    """Tokenize text examples."""
    return student_tokenizer(examples["text"],
                             truncation=True,
                             max_length=config["tokenizer"]["max_length"],
                             padding="max_length")


def prepare_dataset(dataset, student_tokenizer, config):
    """Prepare dataset by formatting and tokenizing."""
    logger.info("Formatting dataset with FreedomIntelligence format...")

    # Format dataset
    dataset = dataset.map(lambda x: freedom_intelligence_format(x, student_tokenizer, config),
                          desc="Formatting FreedomIntelligence dataset")
    logger.info("Dataset formatting complete")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, student_tokenizer, config),
                                    batched=True,
                                    num_proc=8,
                                    remove_columns=["text"])
    logger.info("Tokenization complete")

    # Split into train and test
    logger.info("Splitting dataset into train/test (90/10)...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    logger.info(
        f"Dataset split complete: train={len(tokenized_dataset['train'])}, test={len(tokenized_dataset['test'])}"
    )

    return tokenized_dataset
