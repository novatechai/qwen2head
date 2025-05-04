# config.py

"""Configuration settings for the Dual-Head Qwen3 training."""

# --- Model Configuration ---
model_config = {
    # Base model identifier (ensure this matches the intended Qwen3 version)
    "model_name_or_path": "Qwen/Qwen3-0.6B",
    # Add other model-specific args if any become necessary
    # --- Constants related to model/data structure ---
    "MIMI_CODEBOOK_SIZE": 2048, # Based on Mimi (1st quantizer)
    "MIMI_EXPECTED_SR": 24000, # Sample rate Mimi expects
}

# --- Data Configuration ---
data_config = {
    # Path to the processed dataset folder (Update this after full data prep)
    "data_path": "./data_prepared_FULL/libritts_mimi_codes_1quant",
    # Explicitly define the source Hugging Face dataset ID
    "source_dataset_id": "mythicinfinity/libritts",
    # Name of the training split directory (Update based on full data prep)
    "train_split": "train.clean.360",
    # Optional: Name of the evaluation split directory
    "eval_split": "dev.clean",
    # Column containing the text to be tokenized
    "text_column": "text_normalized",
    # Column containing the target Mimi codes
    "mimi_codes_column": "target_mimi_codes",
    # Moved here from training_config
    "model_max_length": 16384, # Max sequence length for tokenizer and model
}

# --- Loss Configuration ---
# Moved weights here from training_config as they are used by the model's forward pass
loss_config = {
    "loss_weight_text": 1.0, # Weight for the text generation loss
    "loss_weight_mimi": 1.0, # Weight for the Mimi code prediction loss
}

# --- Training Configuration ---
# These settings mirror the arguments in transformers.TrainingArguments
training_config = {
    # --- Core --- 
    "output_dir": "./qwen3-0.6B-dual-head-full-run1", # Directory for checkpoints and final model
    "seed": 42, # Random seed for reproducibility
    "num_train_epochs": 3, # Number of times to iterate over the training dataset
    "per_device_train_batch_size": 8, # Batch size per GPU
    "per_device_eval_batch_size": 8, # Batch size for evaluation
    "gradient_accumulation_steps": 8, # Effective batch size = train_batch_size * num_gpus * grad_accum_steps
    "learning_rate": 1e-5, # Starting learning rate
    "lr_scheduler_type": "cosine", # Learning rate schedule (e.g., linear, cosine)
    "warmup_ratio": 0.03, # Proportion of training steps for linear warmup
    "weight_decay": 0.01, # Weight decay for regularization

    # --- Logging & Saving --- 
    "logging_strategy": "steps", # Log metrics every N steps
    "logging_steps": 10, # Log every 10 steps
    "save_strategy": "steps", # Save checkpoints every N steps
    "save_steps": 500, # Save every 500 steps
    "save_total_limit": 2, # Keep only the last 2 checkpoints
    "report_to": ["wandb"], # Where to report metrics (e.g., "wandb", "tensorboard", "none")
    "run_name": "qwen3-0.6B-dual-head", # Optional name for the run (used by wandb/tensorboard)

    # --- Performance & Precision --- 
    "bf16": True, # Use bfloat16 precision (good for Ampere/Hopper GPUs like H100)
    "tf32": True, # Allow TF32 on Ampere+ GPUs (requires PyTorch >= 1.7)
    "gradient_checkpointing": True, # Save memory by recomputing activations during backward pass
    "dataloader_num_workers": 4, # Number of workers for data loading (adjust based on system)
    "dataloader_pin_memory": True, # Pin memory for faster GPU transfer

    # --- Model & Collator Specific --- 
    "remove_unused_columns": False, # Important: Keep columns needed by collator (target_mimi_codes)
    "label_pad_token_id": -100, # Padding token ID for text labels (ignored by loss)
    "mimi_pad_token_id": -100, # Padding token ID for mimi labels (ignored by loss)
    
    # --- Evaluation --- 
    "eval_strategy": "steps", # Evaluate every N steps
    "eval_steps": 500, # Evaluate every 500 steps (same as save_steps)
    "load_best_model_at_end": True, # Load the best checkpoint based on eval metric at the end
    "metric_for_best_model": "eval_loss", # Metric to determine the best model
    "greater_is_better": False, # For loss, lower is better
    
    # --- Hugging Face Hub Integration --- 
    "push_to_hub": True, # Enable pushing the model to the Hub
    "hub_model_id": "Mazino0/qwen3-2head", # Your target repository ID
    "hub_strategy": "end", # Strategy for pushing ("end", "checkpoint", "all_checkpoints")
    
    # Add any other TrainingArguments you might need...
} 