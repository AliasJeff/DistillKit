"""Evaluation script to compare original student model and distilled student model performance."""

import json
import logging
import torch
import math
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sacrebleu import BLEU
from collections import Counter

from config import CONFIG
from data_processing import load_dataset_split

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, use_flash_attention=True):
    """Load model and tokenizer."""
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception as e:
            logger.warning(f"Flash attention not available: {e}, using default attention")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **model_kwargs)

    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, max_samples=None, stride=512):
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    max_pos = model.config.max_position_embeddings

    with torch.no_grad():
        for sample in tqdm(dataset):
            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).to(device)
            seq_len = input_ids.size(0)

            for i in range(0, seq_len, stride):
                begin = max(i - max_pos, 0)
                end = i + stride
                end = min(end, seq_len)

                input_chunk = input_ids[begin:end].unsqueeze(0)
                labels = input_chunk.clone()

                overlap = i - begin
                labels[:, :overlap] = -100

                outputs = model(input_chunk, labels=labels)
                loss = outputs.loss
                valid_tokens = (labels != -100).sum().item()

                total_nll += loss.item() * valid_tokens
                total_tokens += valid_tokens

    avg_loss = total_nll / total_tokens
    ppl = math.exp(avg_loss)

    return ppl, avg_loss


def compute_model_size(model):
    """Compute model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def compute_bleu(predictions, references):
    """Compute BLEU score.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts (or list of lists for multiple references)
    
    Returns:
        BLEU score (0-100)
    """
    try:
        # Ensure references is in the correct format
        if references and not isinstance(references[0], list):
            references = [[ref] for ref in references]

        bleu = BLEU()
        score = bleu.corpus_score(predictions, references)
        return float(score.score)
    except Exception as e:
        logger.warning(f"Error computing BLEU score: {e}")
        return 0.0


def compute_f1(predictions, references):
    """Compute F1 score based on token overlap.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Average F1 score
    """

    if references and not isinstance(references[0], list):
        references = [[ref] for ref in references]

    def _f1_score(pred_tokens, ref_tokens):
        """Compute F1 score for a single prediction-reference pair."""
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return int(pred_tokens == ref_tokens)

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    try:
        f1_scores = []
        for pred, ref in zip(predictions, references):
            ref_text = ref[0]

            pred_tokens = pred.lower().split()
            ref_tokens = ref_text.lower().split()

            f1 = _f1_score(pred_tokens, ref_tokens)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        return float(avg_f1)
    except Exception as e:
        logger.warning(f"Error computing F1 score: {e}")
        return 0.0


def generate_predictions(model, tokenizer, dataset, max_samples=100, batch_size=8):
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    references = []

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating predictions"):
            batch = dataset[i:i + batch_size]

            prompts = []
            refs = []
            for sample in batch:
                refs.append(sample["Response"])

                messages = [{"role": "user", "content": sample["Question"]}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True).to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for idx in range(len(prompts)):
                input_len = inputs.input_ids[idx].shape[0]
                gen = generated_ids[idx][input_len:]
                pred_text = tokenizer.decode(gen, skip_special_tokens=True)

                predictions.append(pred_text)
                references.append(refs[idx])

    return predictions, references


def evaluate_models(config):  # noqa: C901
    """Main evaluation function."""
    logger.info("Starting model evaluation...")

    # Create results directory
    results_dir = Path(config["training"]["output_dir"]) / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Load test dataset (already formatted with apply_chat_template)
    logger.info("Loading test dataset...")
    test_dataset = load_dataset_split(config, student_tokenizer, split="test")
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")

    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "teacher_model": config["models"]["teacher"],
            "student_model": config["models"]["student"],
            "distillation_temperature": config["distillation"]["temperature"],
            "distillation_alpha": config["distillation"]["alpha"],
        },
        "models": {}
    }

    # Evaluate original student model
    logger.info(f"Evaluating original student model: {config['models']['student']}")
    try:
        student_model, _ = load_model_and_tokenizer(
            config["models"]["student"],
            use_flash_attention=config["model_config"]["use_flash_attention"])

        # Model info
        total_params, trainable_params = count_parameters(student_model)
        model_size = compute_model_size(student_model)

        # Compute perplexity
        logger.info("Computing perplexity for original student model...")
        perplexity, avg_loss = compute_perplexity(student_model,
                                                  student_tokenizer,
                                                  test_dataset,
                                                  max_samples=100)

        # Generate predictions for F1 and BLEU
        logger.info("Generating predictions for F1 and BLEU scores...")
        predictions, references = generate_predictions(student_model,
                                                       student_tokenizer,
                                                       test_dataset,
                                                       max_samples=100)

        # Compute F1 and BLEU
        f1_score = compute_f1(predictions, references)
        bleu_score = compute_bleu(predictions, references)

        results["models"]["original_student"] = {
            "model_name": config["models"]["student"],
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_size_mb": float(model_size),
            "perplexity": float(perplexity),
            "average_loss": float(avg_loss),
            "f1_score": float(f1_score),
            "bleu_score": float(bleu_score),
        }

        logger.info(
            f"Original student model - Perplexity: {perplexity:.4f}, F1: {f1_score:.4f}, BLEU: {bleu_score:.4f}, Size: {model_size:.2f}MB"
        )

        del student_model
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error evaluating original student model: {e}")
        results["models"]["original_student"] = {"error": str(e)}

    # Evaluate distilled student model
    distilled_model_path = config["training"]["output_dir"]
    if Path(distilled_model_path).exists():
        logger.info(f"Evaluating distilled student model: {distilled_model_path}")
        try:
            distilled_model, _ = load_model_and_tokenizer(
                distilled_model_path,
                use_flash_attention=config["model_config"]["use_flash_attention"])

            # Model info
            total_params, trainable_params = count_parameters(distilled_model)
            model_size = compute_model_size(distilled_model)

            # Compute perplexity
            logger.info("Computing perplexity for distilled student model...")
            perplexity, avg_loss = compute_perplexity(distilled_model,
                                                      student_tokenizer,
                                                      test_dataset,
                                                      max_samples=100)

            # Generate predictions for F1 and BLEU
            logger.info("Generating predictions for F1 and BLEU scores...")
            predictions, references = generate_predictions(distilled_model,
                                                           student_tokenizer,
                                                           test_dataset,
                                                           max_samples=100)

            # Compute F1 and BLEU
            f1_score = compute_f1(predictions, references)
            bleu_score = compute_bleu(predictions, references)

            results["models"]["distilled_student"] = {
                "model_path": distilled_model_path,
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "model_size_mb": float(model_size),
                "perplexity": float(perplexity),
                "average_loss": float(avg_loss),
                "f1_score": float(f1_score),
                "bleu_score": float(bleu_score),
            }

            logger.info(
                f"Distilled student model - Perplexity: {perplexity:.4f}, F1: {f1_score:.4f}, BLEU: {bleu_score:.4f}, Size: {model_size:.2f}MB"
            )

            del distilled_model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error evaluating distilled student model: {e}")
            results["models"]["distilled_student"] = {"error": str(e)}
    else:
        logger.warning(f"Distilled model not found at {distilled_model_path}")
        results["models"]["distilled_student"] = {
            "error": f"Model not found at {distilled_model_path}"
        }

    # Compute comparison metrics
    if "original_student" in results["models"] and "distilled_student" in results["models"]:
        if "error" not in results["models"]["original_student"] and "error" not in results[
                "models"]["distilled_student"]:
            original_ppl = results["models"]["original_student"]["perplexity"]
            distilled_ppl = results["models"]["distilled_student"]["perplexity"]

            results["comparison"] = {
                "perplexity_improvement": float(
                    (original_ppl - distilled_ppl) / original_ppl * 100),
                "perplexity_ratio": float(distilled_ppl / original_ppl),
                "original_perplexity": float(original_ppl),
                "distilled_perplexity": float(distilled_ppl),
            }

            logger.info(
                f"Perplexity improvement: {results['comparison']['perplexity_improvement']:.2f}%")

    # Save results
    results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {results_file}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)

    if "original_student" in results["models"] and "error" not in results["models"][
            "original_student"]:
        logger.info("Original Student Model:")
        logger.info(f"  - Perplexity: {results['models']['original_student']['perplexity']:.4f}")
        logger.info(f"  - F1 Score: {results['models']['original_student']['f1_score']:.4f}")
        logger.info(f"  - BLEU Score: {results['models']['original_student']['bleu_score']:.4f}")
        logger.info(
            f"  - Model Size: {results['models']['original_student']['model_size_mb']:.2f}MB")
        logger.info(
            f"  - Parameters: {results['models']['original_student']['total_parameters']:,}")

    if "distilled_student" in results["models"] and "error" not in results["models"][
            "distilled_student"]:
        logger.info("Distilled Student Model:")
        logger.info(f"  - Perplexity: {results['models']['distilled_student']['perplexity']:.4f}")
        logger.info(f"  - F1 Score: {results['models']['distilled_student']['f1_score']:.4f}")
        logger.info(f"  - BLEU Score: {results['models']['distilled_student']['bleu_score']:.4f}")
        logger.info(
            f"  - Model Size: {results['models']['distilled_student']['model_size_mb']:.2f}MB")
        logger.info(
            f"  - Parameters: {results['models']['distilled_student']['total_parameters']:,}")

    if "comparison" in results:
        logger.info("Comparison:")
        logger.info(
            f"  - Perplexity Improvement: {results['comparison']['perplexity_improvement']:.2f}%")
        logger.info(f"  - Perplexity Ratio: {results['comparison']['perplexity_ratio']:.4f}")

    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    results = evaluate_models(CONFIG)
