# DistillLogits

Knowledge distillation framework for training efficient student models using logits-based distillation from teacher models.

## Overview

DistillLogits is a comprehensive knowledge distillation toolkit that enables you to:
- Train efficient student models by distilling knowledge from larger teacher models
- Use logits-based KL-divergence loss for effective knowledge transfer
- Evaluate model performance with multiple metrics (perplexity, BLEU, F1)
- Generate and test model outputs
- Compare original and distilled models

## Features

- **Logits-based Knowledge Distillation**: Uses KL-divergence loss on model logits with configurable temperature and alpha parameters
- **Flexible Model Support**: Works with any HuggingFace transformer model
- **Comprehensive Evaluation**: Computes perplexity, BLEU scores, and F1 scores
- **Model Comparison**: Compare original and distilled models side-by-side
- **Performance Testing**: Generate outputs and measure generation speed
- **Configurable Training**: Supports flash attention, mixed precision, gradient accumulation, and more

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (recommended for GPU acceleration)

## Configuration

Edit `config.py` to customize your training setup

## Usage

### 1. Train the Model

```bash
# Basic training
python main.py train

# Train and evaluate
python main.py train --evaluate

# Full pipeline: train, evaluate, and generate samples
python main.py train --evaluate --generate-samples --num-samples 5
```

### 2. Evaluate Models

Compute perplexity, BLEU, and F1 scores on test dataset:

```bash
python main.py evaluate
```

This will:
- Load original and distilled student models
- Compute perplexity and loss
- Calculate BLEU and F1 scores
- Save results to `results/evaluation/evaluation_results_*.json`

### 3. Test Model Outputs

Generate outputs and measure performance metrics:

```bash
# Test the distilled model
python main.py test

# Test with comparison to original model
python main.py test --compare-original

# Test with custom number of samples
python main.py test --num-samples 10

# Save results to file
python main.py test --compare-original --output-file test_results.json
```

Test metrics include:
- Output length (tokens)
- Generation time per sample
- Average generation time
- Speedup ratio (when comparing models)

### 4. Generate Samples

Generate text from the trained model:

```bash
# Generate with default prompts
python main.py generate

# Generate with custom prompts
python main.py generate --prompts "What is AI?" "Explain machine learning"

# Generate more samples
python main.py generate --num-samples 10
```

### 5. View Configuration

Display or save the current configuration:

```bash
# Display configuration
python main.py config

# Save configuration to file
python main.py config --save config_backup.json
```

## Command Reference

### train
Train the distilled student model.

**Options:**
- `--evaluate`: Run evaluation after training
- `--generate-samples`: Generate samples after training
- `--num-samples N`: Number of samples to generate (default: 5)

**Example:**
```bash
python main.py train --evaluate --generate-samples --num-samples 10
```

### evaluate
Evaluate original and distilled models on test dataset.

**Options:**
- `--max-samples N`: Maximum samples to evaluate (default: 500)

**Output:**
- Perplexity and loss for each model
- BLEU and F1 scores
- Comparison metrics
- Results saved to JSON file

**Example:**
```bash
python main.py evaluate --max-samples 1000
```

### test
Test model outputs with performance metrics.

**Options:**
- `--model-path PATH`: Path to model to test (default: ./results)
- `--compare-original`: Compare with original student model
- `--num-samples N`: Number of test samples (default: 10)
- `--output-file FILE`: Save results to JSON file

**Metrics:**
- Total outputs generated
- Average output length
- Average generation time
- Speedup ratio (when comparing)

**Example:**
```bash
python main.py test --compare-original --num-samples 20 --output-file test_results.json
```

### generate
Generate text samples from the trained model.

**Options:**
- `--model-path PATH`: Path to model (default: ./results)
- `--num-samples N`: Number of samples (default: 5)
- `--prompts TEXT...`: Custom prompts for generation

**Example:**
```bash
python main.py generate --num-samples 5 --prompts "What is AI?" "Explain ML"
```

### config
Display or save configuration.

**Options:**
- `--save FILE`: Save configuration to JSON file

**Example:**
```bash
python main.py config --save my_config.json
```

## Project Structure

```
distill_logits/
├── main.py                 # Main entry point with CLI
├── config.py               # Configuration settings
├── distil_logits.py        # Training logic and custom trainer
├── evaluate.py             # Evaluation functions (perplexity, BLEU, F1)
├── data_processing.py      # Dataset loading and preprocessing
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Components

### LogitsTrainer
Custom trainer class that implements logits-based knowledge distillation:
- Computes KL-divergence loss between student and teacher logits
- Supports logit padding for vocabulary size mismatch
- Combines KD loss with original language modeling loss

### Evaluation Metrics
- **Perplexity**: Measures model's ability to predict test data
- **BLEU Score**: Evaluates n-gram overlap with reference text
- **F1 Score**: Token-level precision and recall

### Data Processing
- Supports FreedomIntelligence dataset format
- Automatic chat template formatting
- Tokenization with configurable max length
- Train/test split (90/10)

## Training Details

### Loss Function
```
Loss = α * KL_Divergence(student_logits, teacher_logits) + (1-α) * CrossEntropy(student_logits, labels)
```

Where:
- `α` (alpha): Balance between distillation and original loss
- `temperature`: Controls softness of probability distributions

### Optimization
- Optimizer: AdamW
- Learning rate scheduler: Cosine annealing
- Gradient accumulation: Supported
- Mixed precision: BF16 support
- Flash Attention 2: Optional for faster computation

## Output Files

### Training
- `results/checkpoint-*/`: Model checkpoints
- `results/pytorch_model.bin`: Final trained model

### Evaluation
- `results/evaluation/evaluation_results_*.json`: Evaluation metrics and comparison

### Testing
- `test_results.json`: Test output metrics (if `--output-file` specified)
