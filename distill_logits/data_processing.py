"""Data processing utilities for distill_logits training."""

import logging
import os
from pathlib import Path
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


def freedom_intelligence_format(example, student_tokenizer, config, mode="train"):
    """Convert FreedomIntelligence format to chat template format."""
    # Question, Complex_CoT, Response
    messages = [{"role": "user", "content": example['Question']}]
    if mode == "train":
        messages.append({"role": "assistant", "content": example['Response']})

    text = student_tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
    return {"text": text}


def tokenize_function(examples, student_tokenizer, config):
    """Tokenize text examples."""
    return student_tokenizer(examples["text"],
                             truncation=True,
                             max_length=config["tokenizer"]["max_length"],
                             padding="max_length")


def get_dataset_cache_dir(config):
    """Get the local cache directory for processed datasets."""
    cache_dir = config.get("dataset_cache_dir", "./dataset_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_split_cache_path(config, split="train"):
    """Get the cache file path for a specific split."""
    cache_dir = get_dataset_cache_dir(config)
    dataset_name = config["dataset"]["name"].replace("/", "_")
    return os.path.join(cache_dir, f"{dataset_name}_{split}")


def load_dataset_split(config, student_tokenizer, split="train"):
    """Load dataset split from cache or process if not cached.
    
    Args:
        config: Configuration dictionary
        student_tokenizer: Tokenizer for the student model
        split: "train" or "test"
    
    Returns:
        The dataset split (either from cache or newly processed)
    """
    split_path = get_split_cache_path(config, split)

    # Check if split exists in cache
    if os.path.exists(split_path):
        logger.info(f"Loading {split} split from cache: {split_path}")
        from datasets import load_from_disk
        return load_from_disk(split_path)

    # If cache doesn't exist, process the full dataset
    logger.info(f"Cache not found for {split} split. Processing dataset...")
    dataset = load_and_preprocess_dataset(config)
    tokenized_dataset = prepare_dataset(dataset, student_tokenizer, config)

    # Return the requested split
    return tokenized_dataset[split]


def prepare_dataset(dataset, student_tokenizer, config, mode="train"):
    """Prepare dataset by formatting and tokenizing.
    
    Saves train and test splits locally for future use.
    """
    logger.info("Formatting dataset with FreedomIntelligence format...")

    # Format dataset
    dataset = dataset.map(lambda x: freedom_intelligence_format(x, student_tokenizer, config, mode),
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
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1,
                                                           seed=config["dataset"]["seed"])
    logger.info(
        f"Dataset split complete: train={len(tokenized_dataset['train'])}, test={len(tokenized_dataset['test'])}"
    )

    # Save splits locally
    cache_dir = get_dataset_cache_dir(config)
    train_path = get_split_cache_path(config, "train")
    test_path = get_split_cache_path(config, "test")

    logger.info(f"Saving train split to {train_path}...")
    tokenized_dataset["train"].save_to_disk(train_path)
    logger.info(f"Saving test split to {test_path}...")
    tokenized_dataset["test"].save_to_disk(test_path)
    logger.info("Dataset splits saved to local cache")

    return tokenized_dataset
