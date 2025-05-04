#!/usr/bin/env python
# coding: utf-8

"""Training script for the Dual-Head Qwen3 model."""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_from_disk

# Import the custom model definition and config
from dual_head_model import Qwen3ForCausalLMWithMimiHead
import config # Import the configuration file

# Define paths and model names (adjust as needed)
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
PROCESSED_DATA_DIR = "./data_prepared_TEST/libritts_mimi_codes_1quant" # Use the TEST dir for now
TRAIN_SPLIT_NAME = "dev.clean" # Use the small test split for initial setup
OUTPUT_MODEL_DIR = "./output_model_TEST"

# --- Implemented Data Collator --- 
@dataclass
class DataCollatorForDualHeadTraining:
    """
    Collates sequences for dual-head training, handling padding for both
    text input_ids/labels and mimi_code_labels.

    Ensures that mimi_code_labels are padded to the same sequence length as input_ids/labels.
    """
    tokenizer: transformers.PreTrainedTokenizer
    label_pad_token_id: int = -100
    mimi_pad_token_id: int = -100
    mimi_codes_column: str = "target_mimi_codes" # Get column name from config later if needed

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract necessary fields. Convert to list first in case they are numpy arrays
        input_ids = [list(instance["input_ids"]) for instance in instances]
        labels = [list(instance["labels"]) for instance in instances]
        
        # Ensure the key exists before trying to access it
        mimi_col = self.mimi_codes_column
        if mimi_col not in instances[0]:
             raise KeyError(f"'{mimi_col}' not found in dataset instance passed to collator. Available keys: "
                           f"{list(instances[0].keys())}")
        mimi_labels_list = [list(instance[mimi_col]) for instance in instances]

        # Use tokenizer.pad to handle input_ids, attention_mask, and text labels padding
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids, "labels": labels},
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )

        # Manually pad the mimi_labels to the same length as the padded input_ids
        max_length = batch_encoding["input_ids"].shape[1]
        padded_mimi_labels = []
        for mimi_seq in mimi_labels_list:
            padding_length = max_length - len(mimi_seq)
            padded_seq = mimi_seq + [self.mimi_pad_token_id] * padding_length
            padded_seq = padded_seq[:max_length]
            padded_mimi_labels.append(padded_seq)

        batch_encoding["mimi_labels"] = torch.tensor(padded_mimi_labels, dtype=torch.long)

        return batch_encoding

def train():
    # --- Use settings from config.py --- 
    model_name_or_path = config.model_config["model_name_or_path"]
    data_path = config.data_config["data_path"]
    train_split = config.data_config["train_split"]
    eval_split = config.data_config.get("eval_split") # Use .get for optional keys
    text_column = config.data_config["text_column"]
    mimi_codes_column = config.data_config["mimi_codes_column"]
    
    # Instantiate TrainingArguments from the dictionary
    training_args = TrainingArguments(**config.training_config)
    # ------------------------------------

    # Set seed using the value from training_args
    set_seed(training_args.seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, # Use var from config
        model_max_length=config.data_config["model_max_length"], # Use var from config
        padding_side="right", 
        use_fast=False, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad token, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = Qwen3ForCausalLMWithMimiHead.from_pretrained(
        model_name_or_path, # Use var from config
        trust_remote_code=True,
        ignore_mismatched_sizes=True, 
        # Pass torch_dtype if specified in training_args (e.g., for bf16)
        # torch_dtype=torch.bfloat16 if training_args.bf16 else None, 
        # device_map="auto" # Consider adding for multi-GPU
    )
    print("Model loaded.")

    # Load train dataset
    print(f"Loading train dataset from {data_path}/{train_split}...")
    train_dataset = load_from_disk(os.path.join(data_path, train_split))
    print(f"Train dataset loaded. Number of samples: {len(train_dataset)}")
    print(f"Original train dataset columns: {train_dataset.column_names}")
    
    # Load evaluation dataset (optional)
    eval_dataset = None
    if eval_split and training_args.do_eval:
        eval_path = os.path.join(data_path, eval_split)
        print(f"Loading eval dataset from {eval_path}...")
        if os.path.exists(eval_path):
            eval_dataset = load_from_disk(eval_path)
            print(f"Eval dataset loaded. Number of samples: {len(eval_dataset)}")
            print(f"Original eval dataset columns: {eval_dataset.column_names}")
        else:
            print(f"WARNING: Evaluation split directory not found: {eval_path}")
            eval_dataset = None

    # --- Dataset Preprocessing Function --- 
    def preprocess_function(examples):
        tokenized_outputs = tokenizer(
            examples[text_column], # Use var from config
            max_length=config.data_config["model_max_length"], # Use var from config
            truncation=True,
            padding=False, 
        )
        tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()
        tokenized_outputs[mimi_codes_column] = examples[mimi_codes_column] # Use var from config
        return tokenized_outputs
    # ----------------------------------------

    print("Preprocessing train dataset...")
    original_cols_train = list(train_dataset.column_names)
    cols_to_remove_train = [col for col in original_cols_train if col != mimi_codes_column]
    print(f"Train columns to remove: {cols_to_remove_train}")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=training_args.dataloader_num_workers, # Use num_workers from config
        remove_columns=cols_to_remove_train 
    )
    print(f"Train dataset preprocessed. New columns: {train_dataset.column_names}")
    
    # Preprocess eval dataset if it exists
    if eval_dataset:
        print("Preprocessing eval dataset...")
        original_cols_eval = list(eval_dataset.column_names)
        cols_to_remove_eval = [col for col in original_cols_eval if col != mimi_codes_column]
        print(f"Eval columns to remove: {cols_to_remove_eval}")
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataloader_num_workers,
            remove_columns=cols_to_remove_eval
        )
        print(f"Eval dataset preprocessed. New columns: {eval_dataset.column_names}")

    # Initialize data collator
    data_collator = DataCollatorForDualHeadTraining(
        tokenizer=tokenizer,
        label_pad_token_id=training_args.label_pad_token_id,
        mimi_pad_token_id=training_args.mimi_pad_token_id,
        mimi_codes_column=mimi_codes_column # Pass column name
    )
    print("Data collator initialized.")

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass eval dataset
        data_collator=data_collator,
        # Add compute_metrics function here if needed for evaluation
    )
    print("Trainer initialized.")

    # Start training
    print("Starting training...")
    try:
        train_result = trainer.train()
        print("Training finished successfully.")
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        # Potentially save state or model even if training failed partially
        # trainer.save_model(os.path.join(training_args.output_dir, "failed_checkpoint"))

    # Save the final model (only if training didn't error out early)
    if trainer.state.is_world_process_zero: # Ensure only main process saves
        print("Saving final model...")
        trainer.save_model(training_args.output_dir)
        # tokenizer.save_pretrained(training_args.output_dir) # Save tokenizer too
        print(f"Model saved to {training_args.output_dir}")
        
    # Optional: Evaluate after training if eval dataset provided
    if eval_dataset and training_args.do_eval:
        print("*** Evaluate ***")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

if __name__ == "__main__":
    train() 
    train() 