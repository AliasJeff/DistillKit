"""Main entry point for distill_logits training and evaluation."""

import argparse
import logging
import sys

from config import CONFIG
from distil_logits import main as train_main
from evaluate import evaluate_models

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Distill Logits: Knowledge Distillation for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train

  # Evaluate models
  python main.py evaluate

  # Train and then evaluate
  python main.py train --evaluate

  # Show configuration
  python main.py config

  # Launch Gradio web interface
  python main.py gradio

  # Full pipeline: train, evaluate, and generate samples
  python main.py train --evaluate --generate-samples
        """)

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the distilled model')
    train_parser.add_argument('--evaluate',
                              action='store_true',
                              help='Run evaluation after training')
    train_parser.add_argument('--generate-samples',
                              action='store_true',
                              help='Generate sample outputs after training')
    train_parser.add_argument('--num-samples',
                              type=int,
                              default=5,
                              help='Number of samples to generate')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--max-samples',
                             type=int,
                             default=500,
                             help='Maximum samples to evaluate')

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--save', type=str, help='Save configuration to file')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate samples from model')
    gen_parser.add_argument('--model-path',
                            type=str,
                            default=CONFIG["training"]["output_dir"],
                            help='Path to model')
    gen_parser.add_argument('--num-samples',
                            type=int,
                            default=5,
                            help='Number of samples to generate')
    gen_parser.add_argument('--prompts', type=str, nargs='+', help='Prompts for generation')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test model outputs with metrics')
    test_parser.add_argument('--model-path',
                             type=str,
                             default=CONFIG["training"]["output_dir"],
                             help='Path to model to test')
    test_parser.add_argument('--compare-original',
                             action='store_true',
                             help='Compare with original student model')
    test_parser.add_argument('--num-samples', type=int, default=10, help='Number of test samples')
    test_parser.add_argument('--output-file', type=str, help='Save test results to file')

    # Gradio command
    gradio_parser = subparsers.add_parser('gradio', help='Launch Gradio web interface')

    return parser


def show_config(config, save_path=None):
    """Display and optionally save configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)

    import json
    config_str = json.dumps(config, indent=2)
    logger.info(config_str)
    logger.info("=" * 60 + "\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {save_path}")


def train_command(args):
    """Execute training command."""
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {CONFIG['project_name']}")

    try:
        # Run training
        train_main()
        logger.info("Training completed successfully!")

        # Run evaluation if requested
        if args.evaluate:
            logger.info("\nStarting evaluation...")
            evaluate_models(CONFIG)
            logger.info("Evaluation completed successfully!")

            # Generate samples if requested
            if args.generate_samples:
                logger.info("\nGenerating sample outputs...")
                generate_samples(CONFIG, num_samples=args.num_samples)

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


def evaluate_command(args):
    """Execute evaluation command."""
    logger.info("Starting evaluation...")

    try:
        results = evaluate_models(CONFIG)
        logger.info("Evaluation completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)


def config_command(args):
    """Execute config command."""
    show_config(CONFIG, save_path=args.save)


def generate_samples(config, num_samples=5, prompts=None):
    """Generate sample outputs from the trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data_processing import load_dataset_split
    import torch
    import random

    logger.info(f"Loading model from {config['training']['output_dir']}")

    try:
        # Load model and tokenizer
        model_path = config["training"]["output_dir"]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.chat_template = config["tokenizer"]["chat_template"]
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto")

        # Load prompts from dataset if none provided
        if not prompts:
            logger.info("Loading prompts from dataset...")
            test_dataset = load_dataset_split(config, tokenizer, split="test")
            # Randomly sample questions from dataset
            num_to_sample = min(num_samples, len(test_dataset))
            sample_indices = random.sample(range(len(test_dataset)), num_to_sample)
            prompts = [test_dataset[idx]["Question"] for idx in sample_indices]
            logger.info(f"Loaded {len(prompts)} prompts from dataset")

        logger.info(f"\nGenerating {len(prompts)} samples...\n")
        logger.info("=" * 60)

        model.eval()
        with torch.no_grad():
            for i, question in enumerate(prompts, 1):
                logger.info(f"Sample {i}:")
                logger.info(f"Question: {question}")

                # Use apply_chat_template to format the prompt
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(messages,
                                                       tokenize=False,
                                                       add_generation_prompt=True)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                generated_ids = model.generate(**inputs,
                                               max_new_tokens=256,
                                               num_beams=1,
                                               temperature=0.7,
                                               top_p=0.9,
                                               do_sample=True,
                                               eos_token_id=tokenizer.eos_token_id,
                                               pad_token_id=tokenizer.eos_token_id)

                # Decode only the generated part
                generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:],
                                                  skip_special_tokens=True)
                logger.info(f"Output: {generated_text}\n")

        logger.info("=" * 60)
        logger.info("Sample generation completed!")

    except Exception as e:
        logger.error(f"Error generating samples: {e}", exc_info=True)
        sys.exit(1)


def generate_command(args):
    """Execute generate command."""
    logger.info("Starting sample generation...")

    try:
        generate_samples(CONFIG, num_samples=args.num_samples, prompts=args.prompts)

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)


def test_model_outputs(  # noqa: C901
        model_path,
        num_samples=10,
        compare_original=False,
        output_file=None,
):
    """Test model outputs with various metrics.
    
    Args:
        model_path: Path to the model to test
        num_samples: Number of test samples to generate
        compare_original: Whether to compare with original student model
        output_file: Optional file to save results (auto-generated if not provided)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import json
    from datetime import datetime
    from data_processing import load_dataset_split

    logger.info("\n" + "=" * 70)
    logger.info("MODEL OUTPUT TEST")
    logger.info("=" * 70)

    # Generate default output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.json"

    test_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "num_samples": num_samples,
        "models": {}
    }

    # Load test prompts from dataset
    logger.info("Loading test prompts from dataset...")
    # Create a temporary tokenizer for loading the dataset
    temp_tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["student"])
    temp_tokenizer.chat_template = CONFIG["tokenizer"]["chat_template"]
    test_dataset = load_dataset_split(CONFIG, temp_tokenizer, split="test")
    # Extract questions from the dataset
    test_prompts = []
    for i in range(min(num_samples, len(test_dataset))):
        test_prompts.append(test_dataset[i]['Question'])

    logger.info(f"Loaded {len(test_prompts)} test prompts from dataset")

    def test_single_model(model_name, model_path_to_load):
        """Test a single model."""
        logger.info(f"\nTesting model: {model_name}")
        logger.info(f"Model path: {model_path_to_load}")

        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path_to_load)
            tokenizer.chat_template = CONFIG["tokenizer"]["chat_template"]
            model = AutoModelForCausalLM.from_pretrained(model_path_to_load,
                                                         torch_dtype=torch.bfloat16,
                                                         device_map="auto")

            model.eval()
            outputs = []
            total_time = 0

            logger.info(f"Generating {len(test_prompts)} test outputs...")

            with torch.no_grad():
                for i, question in enumerate(test_prompts, 1):
                    import time
                    start_time = time.time()

                    # Use apply_chat_template to format the prompt
                    messages = [{"role": "user", "content": question}]
                    prompt = tokenizer.apply_chat_template(messages,
                                                           tokenize=False,
                                                           add_generation_prompt=True)

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    generated_ids = model.generate(**inputs,
                                                   max_new_tokens=256,
                                                   num_beams=1,
                                                   temperature=0.7,
                                                   top_p=0.9,
                                                   do_sample=True,
                                                   eos_token_id=tokenizer.eos_token_id,
                                                   pad_token_id=tokenizer.eos_token_id)

                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time

                    # Decode only the generated part
                    generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:],
                                                      skip_special_tokens=True)

                    output_info = {
                        "question": question,
                        "output": generated_text,
                        "output_length": len(generated_text.split()),
                        "generation_time": elapsed_time
                    }
                    outputs.append(output_info)

                    logger.info(
                        f"  [{i}/{len(test_prompts)}] Generated {len(generated_text.split())} tokens in {elapsed_time:.2f}s"
                    )

            # Calculate statistics
            avg_output_length = sum(o["output_length"] for o in outputs) / len(outputs)
            avg_time = total_time / len(outputs)

            model_results = {
                "model_name": model_name,
                "total_outputs": len(outputs),
                "average_output_length": float(avg_output_length),
                "average_generation_time": float(avg_time),
                "total_generation_time": float(total_time),
                "outputs": outputs
            }

            logger.info(f"\n{model_name} Statistics:")
            logger.info(f"  - Total outputs: {len(outputs)}")
            logger.info(f"  - Average output length: {avg_output_length:.1f} tokens")
            logger.info(f"  - Average generation time: {avg_time:.3f}s")
            logger.info(f"  - Total generation time: {total_time:.2f}s")

            del model
            torch.cuda.empty_cache()

            return model_results

        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}", exc_info=True)
            return {"model_name": model_name, "error": str(e)}

    # Test the specified model
    test_results["models"]["target_model"] = test_single_model("Target Model", model_path)

    # Optionally compare with original student model and teacher model
    if compare_original:
        logger.info("\n" + "-" * 70)
        original_model_path = CONFIG["models"]["student"]
        test_results["models"]["original_student"] = test_single_model(
            "Original Student Model", original_model_path)

        # Also test teacher model
        logger.info("\n" + "-" * 70)
        teacher_model_path = CONFIG["models"]["teacher"]
        test_results["models"]["teacher"] = test_single_model("Teacher Model", teacher_model_path)

        # Compare metrics
        target_results = test_results["models"]["target_model"]
        original_results = test_results["models"]["original_student"]
        teacher_results = test_results["models"]["teacher"]

        if ("error" not in target_results and "error" not in original_results
                and "error" not in teacher_results):
            target_time = target_results["average_generation_time"]
            original_time = original_results["average_generation_time"]
            teacher_time = teacher_results["average_generation_time"]

            speedup_vs_original = original_time / target_time if target_time > 0 else 0
            speedup_vs_teacher = teacher_time / target_time if target_time > 0 else 0

            test_results["comparison"] = {
                "target_avg_time": float(target_time),
                "original_student_avg_time": float(original_time),
                "teacher_avg_time": float(teacher_time),
                "speedup_vs_original": float(speedup_vs_original),
                "speedup_vs_teacher": float(speedup_vs_teacher)
            }

            logger.info("\n" + "-" * 70)
            logger.info("Comparison Results:")
            logger.info(f"  - Target model avg time: {target_time:.3f}s")
            logger.info(f"  - Original student model avg time: {original_time:.3f}s")
            logger.info(f"  - Teacher model avg time: {teacher_time:.3f}s")
            logger.info("\nSpeedup ratios:")
            logger.info(f"  - Target vs Original Student: {speedup_vs_original:.2f}x")
            logger.info(f"  - Target vs Teacher: {speedup_vs_teacher:.2f}x")

    # Save results
    try:
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"\nTest results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving test results to {output_file}: {e}")

    logger.info("=" * 70 + "\n")

    return test_results


def test_command(args):
    """Execute test command."""
    logger.info("Starting model output test...")

    try:
        results = test_model_outputs(model_path=args.model_path,
                                     num_samples=args.num_samples,
                                     compare_original=args.compare_original,
                                     output_file=args.output_file)
        logger.info("Model output test completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        sys.exit(1)


def gradio_command(args):
    """Launch the Gradio web interface."""
    logger.info("Launching Gradio web interface for model comparison...")
    try:
        from gradio_ui import main as gradio_main
        gradio_main()
    except ImportError as e:
        logger.error("Could not import Gradio. Please install it with 'pip install gradio'")
        logger.error(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return

    # Route to appropriate command handler
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'config':
        config_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'gradio':
        gradio_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
