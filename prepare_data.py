#!/usr/bin/env python
# coding: utf-8

"""Script to prepare LibriTTS data with Mimi audio codes."""

import datasets
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import os
import itertools
# Add necessary imports for Features and Generator
from datasets import Features, Value, Sequence
from datasets import Dataset # Import Dataset class

# Import config
import config

# Constants
# SOURCE_DATASET = config.data_config["data_path"].split('/')[1] # Infer from data_path -- REMOVED, using config directly
# Use TARGET_DATASET_NAME based on config.data_config["data_path"]
TARGET_DATASET_NAME = os.path.basename(config.data_config["data_path"])
# --- Get constants from config --- 
MIMI_EXPECTED_SR = config.model_config["MIMI_EXPECTED_SR"]
# ---------------------------------
# SPLITS_TO_PROCESS = ["dev.clean"] # Define which splits to process (e.g., [config.data_config["train_split"], config.data_config["eval_split"]])
# Process train and eval splits defined in config
SPLITS_TO_PROCESS = [config.data_config["train_split"]]
if "eval_split" in config.data_config and config.data_config["eval_split"]:
    SPLITS_TO_PROCESS.append(config.data_config["eval_split"])

# NUM_TEST_SAMPLES = 10 # Remove this for full processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# OUTPUT_DIR = "./data_prepared_TEST" # Use path from config
OUTPUT_DIR = os.path.dirname(config.data_config["data_path"]) # Get directory from data_path


# Global variable for the model to be accessible in map function
mimi_model = None

def add_mimi_codes(batch):
    """Processes a batch of audio data to add Mimi codes."""
    global mimi_model
    global MIMI_EXPECTED_SR # Use the config value
    if mimi_model is None:
        raise RuntimeError("Mimi model not loaded")

    is_batched = isinstance(batch["audio"], list)

    if is_batched:
        audio_list = batch["audio"]
    else:
        audio_list = [batch["audio"]]
        for key in batch.keys():
            if key != "audio" and isinstance(batch[key], (str, int, float)):
                 batch[key] = [batch[key]]
            elif key != "audio" and isinstance(batch[key], np.ndarray):
                 batch[key] = [batch[key].tolist()]

    processed_audios = []
    max_len = 0

    for audio_item in audio_list:
        waveform = torch.tensor(audio_item["array"], dtype=torch.float32)
        sr = audio_item["sampling_rate"]

        if sr != MIMI_EXPECTED_SR:
            waveform_device = waveform.to(DEVICE)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=MIMI_EXPECTED_SR).to(DEVICE)
            waveform = resampler(waveform_device)
        else:
            waveform = waveform.to(DEVICE)

        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                 waveform = waveform[0, :] 
                 waveform = waveform.unsqueeze(0) 
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
             raise ValueError(f"Unexpected waveform dimension: {waveform.dim()}")

        waveform = waveform.unsqueeze(1) 
        processed_audios.append(waveform)
        max_len = max(max_len, waveform.shape[-1])

    if is_batched and len(processed_audios) > 1:
        padded_audios = []
        for waveform in processed_audios:
            padding_needed = max_len - waveform.shape[-1]
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            padded_audios.append(padded_waveform)
        audio_batch_tensor = torch.cat(padded_audios, dim=0).to(DEVICE)
    elif len(processed_audios) == 1:
        audio_batch_tensor = processed_audios[0].to(DEVICE)
    else:
        return {}

    with torch.no_grad():
        input_tensor = audio_batch_tensor
        if not is_batched:
             pass
        codes = mimi_model.encode(input_tensor)

    first_quantizer_codes = codes[:, 0, :].cpu().numpy()

    # Use column name from config
    text_col = config.data_config["text_column"]
    if text_col in batch:
         batch[text_col] = batch[text_col]
    else: # Fallback or error if primary text column not found
         print(f"Warning: Specified text column '{text_col}' not found in batch. Keys: {list(batch.keys())}")
         # Decide how to handle this - maybe try 'text_original' or raise error
         if "text_original" in batch:
             batch["text_original"] = batch["text_original"]
         elif "text" in batch:
              batch["text"] = batch["text"]

    if not is_batched:
        batch[config.data_config["mimi_codes_column"]] = first_quantizer_codes[0].tolist()
    else:
        batch[config.data_config["mimi_codes_column"]] = first_quantizer_codes.tolist()

    if not is_batched:
        for key in batch.keys():
            if isinstance(batch[key], list) and len(batch[key]) == 1 and key != config.data_config["mimi_codes_column"]:
                 batch[key] = batch[key][0]

    return batch

def main():
    global mimi_model
    global MIMI_EXPECTED_SR # Use the config value
    
    print("Loading Mimi model...")
    try:
        mimi_weight_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi_model = loaders.get_mimi(mimi_weight_path, device=DEVICE)
        print(f"Successfully loaded Mimi model to {DEVICE}.")
        print(f"Mimi sample rate: {mimi_model.sample_rate} Hz")
        if mimi_model.sample_rate != MIMI_EXPECTED_SR:
             raise ValueError(f"Mimi model sample rate {mimi_model.sample_rate} != expected {MIMI_EXPECTED_SR}")

    except Exception as e:
        print(f"ERROR loading Mimi model: {e}")
        return

    # --- Define the features for the output dataset --- 
    final_features = Features({
        config.data_config["text_column"]: Value("string"),
        config.data_config["mimi_codes_column"]: Sequence(Value("int64"))
    })
    print(f"Target features defined: {final_features}")
    # ---------------------------------------------------

    # Ensure base output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in SPLITS_TO_PROCESS:
        if not split:
            continue # Skip if eval_split is None/empty in config
        
        print(f"\nProcessing split: {split}")
        print("Loading dataset (might take time for full split)...")
        # Load the full dataset split (not streaming)
        try:
            # --- Use the explicit source_dataset_id from config --- 
            source_dataset_id = config.data_config["source_dataset_id"]
            # -------------------------------------------------------
            print(f"Attempting to load source dataset: {source_dataset_id} with split: {split} (Streaming)")
            # --- Load dataset without specifying a 'name', as Elise uses a default config with a 'train' split --- 
            ds_stream = load_dataset(source_dataset_id, split=split, trust_remote_code=True, streaming=True)
        except Exception as e:
            # If loading fails, print error and continue
            print(f"ERROR loading dataset '{source_dataset_id}' with split '{split}'. Error: {e}")
            # Optional: Uncomment to try 'all' as a fallback if 'clean' fails
            # print(f"Trying fallback to name='all'...")
            # try:
            #     ds_stream = load_dataset(source_dataset_id, name="all", split=split, trust_remote_code=True, streaming=True)
            # except Exception as e2:
            #     print(f"ERROR loading dataset '{source_dataset_id}' with fallback name='all'. Error: {e2}")
            #     continue # Skip to next split if loading fails
            continue # Skip to next split if loading fails

        # --- Apply the mapping function to the stream --- 
        print("Generating Mimi codes from stream (will process on the fly)...")
        
        # Map the function - remove num_proc and remove_columns for streaming
        processed_stream = ds_stream.map(
            add_mimi_codes,
            batched=True,
            batch_size=4, # <-- Reduced batch size
            # remove_columns=[col for col in list(ds_stream.features.keys()) if col != config.data_config["text_column"] and col != "audio"] # remove_columns is less reliable in streaming
            # num_proc=num_workers # num_proc often problematic with streaming map
        )
        
        # --- Define the generator function --- 
        def processed_generator(stream):
            print("Starting generator to yield processed samples...")
            count = 0
            text_col = config.data_config["text_column"]
            mimi_col = config.data_config["mimi_codes_column"]
            for sample in stream:
                 # Ensure the required columns exist after the map function
                print(f"DEBUG: Sample keys: {list(sample.keys())}, Expected text_col: '{text_col}', Expected mimi_col: '{mimi_col}'")
                if text_col not in sample or mimi_col not in sample:
                    print(f"Warning: Skipping sample due to missing columns. Found keys: {list(sample.keys())}")
                    continue
                yield {
                    text_col: sample[text_col],
                    mimi_col: sample[mimi_col]
                }
                count += 1
                if count % 1000 == 0:
                     print(f"Generator yielded {count} samples...")
            print(f"Generator finished. Total samples yielded: {count}")
        # -------------------------------------
        
        # --- Create Dataset from generator --- 
        print("Creating final Dataset using from_generator...")
        # Pass the generator function itself (as a callable via lambda)
        final_dataset = Dataset.from_generator(lambda: processed_generator(processed_stream), features=final_features)
        print(f"Final Dataset created. Features: {final_dataset.features}")
        # -------------------------------------

        # --- Save the processed dataset --- 
        split_output_dir = os.path.join(OUTPUT_DIR, TARGET_DATASET_NAME, split)
        print(f"Saving processed dataset to: {split_output_dir}")
        # Save the dataset created from the generator
        final_dataset.save_to_disk(split_output_dir)
        print(f"Data for split '{split}' saved.")

    print(f"\nFull Data preparation finished. Processed data saved in {OUTPUT_DIR}/{TARGET_DATASET_NAME}")

if __name__ == "__main__":
    main() 