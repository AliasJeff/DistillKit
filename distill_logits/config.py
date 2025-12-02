"""Configuration for distill_logits training."""

CONFIG = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "split": "train",
        # "num_samples": , # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "XformAI-india/Qwen3-1.7B-medicaldataset",
        "student": "Qwen/Qwen3-0.6B"
    },
    "tokenizer": {
        "max_length":
        4096,
        "chat_template":
        "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint":
        None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}
