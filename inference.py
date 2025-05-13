#!/usr/bin/env python
# coding: utf-8

"""Inference script for the Dual-Head Qwen3 model using interleaved generation."""

import torch
import torchaudio
import transformers
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from moshi.models import loaders # For Mimi decoder
import argparse
import time
import os
from typing import List, Tuple # Added List, Tuple
import torch.nn.functional as F # Added for sampling
import soundfile as sf
import numpy as np

# Import the custom model definition
from dual_head_model import Qwen3ForCausalLMWithMimiHead

# Import config
import config

# --- Configuration --- 
# Use paths and constants from config.py
MODEL_PATH = config.training_config["output_dir"] # Path to the fine-tuned model directory
TOKENIZER_PATH = config.model_config["model_name_or_path"] # Base model for tokenizer
MIMI_EXPECTED_SR = config.model_config["MIMI_EXPECTED_SR"]
DEFAULT_PROMPT = "/no_think Hi whats up?"
OUTPUT_FILENAME = "generated_audio_refined_interleaved.wav"
MAX_NEW_TEXT_TOKENS = 50
CODES_PER_STEP = 4 # Number of Mimi codes per text token
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95 # Set to a value like 0.95 for nucleus sampling
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions --- 

def load_models(model_path: str, device: str):
    """Loads the fine-tuned dual-head model, tokenizer, and Mimi decoder."""
    print(f"Loading models from path: {model_path}...")
    
    # Load Tokenizer (should match the one used for training)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        raise

    # Load Dual-Head Model
    try:
        # Load the custom configuration if saved, otherwise base config
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        print(f"Loading fine-tuned dual-head model from: {model_path}")
        # Ensure the custom class is registered if needed, or load directly
        # We need to explicitly pass the config to our custom class
        model = Qwen3ForCausalLMWithMimiHead.from_pretrained(
            model_path,
            config=model_config, # Pass the loaded config
            trust_remote_code=True, # Might still be needed depending on base model
            # torch_dtype=torch.bfloat16 # Use appropriate dtype
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully.")

        print(f"Loading tokenizer from: {TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Set pad token if not defined (common issue)
            # Using eos_token is a common practice, but check if Qwen3 has a specific pad token
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer pad_token to eos_token ({tokenizer.pad_token})")
        tokenizer.padding_side = 'left' # Important for decoder-only models

        print("Loading Mimi decoder model...")
        mimi_weight_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        # Load Mimi to the same device as the main model
        mimi_decoder = loaders.get_mimi(mimi_weight_path, device=device)
        print("Mimi decoder loaded.")

        return model, tokenizer, mimi_decoder

    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def decode_mimi_codes(decoder_model, codes_tensor: torch.Tensor, sample_rate: int, output_path: str):
    """Decodes a tensor of Mimi codes (1st quantizer) back to a WAV file."""
    global DEVICE # Access the global device variable
    if not isinstance(codes_tensor, torch.Tensor):
         codes_tensor = torch.tensor(codes_tensor, dtype=torch.long)
         
    if codes_tensor.dim() == 1: # If it's a flat list of codes [T_codes]
        codes_tensor = codes_tensor.unsqueeze(0) # Add batch dim -> [1, T_codes]
    
    # --- Start of modification for 8 codebooks ---
    num_expected_codebooks = 8
    batch_size, num_codes_generated = codes_tensor.shape

    # Assume the generated codes_tensor is for the first codebook
    first_codebook = codes_tensor.unsqueeze(1).to(DEVICE)  # Shape: [B, 1, T_codes]

    if num_expected_codebooks > 1:
        # Create padding for the remaining codebooks
        # Using 0 as a neutral padding value.
        padding_shape = (batch_size, num_expected_codebooks - 1, num_codes_generated)
        padding_codes = torch.zeros(padding_shape, dtype=first_codebook.dtype, device=DEVICE)
        
        # Concatenate the first codebook with the padding
        codes_for_decoder = torch.cat([first_codebook, padding_codes], dim=1)
    else:
        # If for some reason only 1 codebook was expected, use the original logic
        codes_for_decoder = first_codebook
    # --- End of modification ---
    
    # Mimi decode expects [B, K, T_codes]. We only have K=1 (1st quantizer)
    # We need to reshape/unsqueeze to [B=1, K=1, T_codes]
    # Move tensor to the correct device using the global DEVICE variable
    # codes_tensor = codes_tensor.unsqueeze(1).to(DEVICE) # This line is now replaced by the logic above
    
    print(f"Decoding Mimi codes with shape: {codes_for_decoder.shape} on device {DEVICE}...")
    with torch.no_grad():
        # TODO: Verify if mimi.decode works with only the first codebook channel.
        # It might expect all codebooks. If so, we need to pad with dummy codes
        # for the other codebooks or adjust the decoder call if possible.
        # Assuming for now it works or we need to adapt.
        try:
            # Ensure decoder_model is used (it should already be on the correct device)
            decoded_waveform = decoder_model.decode(codes_for_decoder) # Use codes_for_decoder
            print(f"Decoded waveform shape: {decoded_waveform.shape}") # Expect [B, 1, T_audio]
            
            # --- Correction: Remove only batch dimension, keep channel --- 
            decoded_waveform = decoded_waveform.squeeze(0).cpu() # Shape becomes [1, T_audio]
            print(f"Shape after squeeze(0): {decoded_waveform.shape}")

            print(f"Saving decoded audio to {output_path} with SR {sample_rate}...")
            torchaudio.save(output_path, decoded_waveform.float(), sample_rate) # Ensure float for torchaudio save
            print("Audio saved.")
        except Exception as e:
            print(f"ERROR during Mimi decoding or saving: {e}")
            print("Proceeding without saving audio.")

# --- Interleaved Generation Logic (Heuristic Implementation) --- 

def generate_interleaved(
    model: Qwen3ForCausalLMWithMimiHead, 
    tokenizer: transformers.PreTrainedTokenizer, 
    prompt: str, 
    max_new_text_tokens: int = MAX_NEW_TEXT_TOKENS, 
    codes_per_step: int = CODES_PER_STEP,
    temperature: float = TEMPERATURE, 
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    **kwargs # Catch other potential kwargs
) -> Tuple[str, List[int]]:
    """
    Generates text and interleaved Mimi codes autoregressively.
    Mimi codes within each step are also generated autoregressively.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    generated_text_ids = prompt_ids.clone()
    generated_mimi_ids = []
    current_past_key_values = None

    print("\nStarting generation...")
    print(f"Prompt: '{prompt}'")
    print(f"Max new text tokens: {max_new_text_tokens}, Codes per step: {codes_per_step}")
    print(f"Sampling: Temp={temperature}, Top-K={top_k}, Top-P={top_p}")

    with torch.no_grad():
        for i in range(max_new_text_tokens):
            # --- Text Token Generation ---
            if current_past_key_values is None:
                # First step: use the whole prompt
                input_ids_step = generated_text_ids
            else:
                # Subsequent steps: use only the last generated text token
                input_ids_step = generated_text_ids[:, -1:]

            # Create attention mask dynamically
            attention_mask = torch.ones_like(generated_text_ids) # Mask includes prompt + generated text

            outputs = model(
                input_ids=input_ids_step,
                attention_mask=attention_mask if current_past_key_values is None else None, # Only need mask for first step w/o PKV
                past_key_values=current_past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False # Not needed for basic generation
            )

            # Get logits for the very last token position
            next_token_logits = outputs['logits'][:, -1, :]

            # Sample the next text token
            next_text_token_id = sample_from_logits(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            # Append generated text token
            generated_text_ids = torch.cat([generated_text_ids, next_text_token_id], dim=-1)

            # Update past_key_values for the next step (text or mimi)
            current_past_key_values = outputs['past_key_values']

            # Check for EOS token
            if next_text_token_id.item() == tokenizer.eos_token_id:
                print("\nEOS token generated (text). Stopping.")
                break

            # --- Mimi Code Generation (Autoregressive Sub-Loop) ---
            step_mimi_ids = []
            # Use the hidden state that predicted the text token to predict the FIRST mimi code
            # The mimi_code_logits are calculated in the forward pass alongside text logits
            mimi_logits_step_0 = outputs['mimi_code_logits'][:, -1, :]

            # Sample first mimi token for this step
            current_mimi_token_id = sample_from_logits(mimi_logits_step_0, temperature=temperature, top_k=top_k, top_p=top_p)
            step_mimi_ids.append(current_mimi_token_id.item())

            # Generate remaining mimi tokens for this step autoregressively
            mimi_past_key_values_sub = current_past_key_values # Start with PKV from text step
            mimi_input_ids_sub = current_mimi_token_id # Input for the *next* mimi token prediction

            for _ in range(1, codes_per_step):
                mimi_outputs = model(
                    input_ids=mimi_input_ids_sub,
                    attention_mask=None, # Not needed when using PKV for single token input
                    past_key_values=mimi_past_key_values_sub,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False
                )

                # Get Mimi logits for the next position
                next_mimi_logits = mimi_outputs['mimi_code_logits'][:, -1, :]
                mimi_past_key_values_sub = mimi_outputs['past_key_values'] # Update PKV *within* mimi sub-loop

                # Sample the next mimi token
                next_mimi_token_id = sample_from_logits(next_mimi_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                step_mimi_ids.append(next_mimi_token_id.item())

                # Update input for the next mimi token prediction
                mimi_input_ids_sub = next_mimi_token_id

            # Append the generated codes for this step
            generated_mimi_ids.extend(step_mimi_ids)

            # Update the main past_key_values with the state after the mimi sub-loop
            current_past_key_values = mimi_past_key_values_sub

            # Print progress periodically
            if (i + 1) % 10 == 0:
                 decoded_text = tokenizer.decode(generated_text_ids[0, prompt_ids.shape[-1]:], skip_special_tokens=True)
                 print(f"  Step {i+1}: Text token {next_text_token_id.item()} -> ...'{decoded_text[-30:]}'")
                 print(f"           Mimi codes this step: {step_mimi_ids}")

    # Decode final results
    decoded_text = tokenizer.decode(generated_text_ids[0, prompt_ids.shape[-1]:], skip_special_tokens=True) # Exclude prompt
    # generated_mimi_ids is already a list of ints

    print("\nGeneration finished.")
    print(f"Generated Text: '{decoded_text}'")
    print(f"Total Mimi Codes generated: {len(generated_mimi_ids)}")
    # print(f"First 50 Mimi Codes: {generated_mimi_ids[:50]}") # Optional: print some codes

    return decoded_text, generated_mimi_ids

# --- Helper Function: Sampling ---
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Samples a token ID from logits using temperature, top-k, and top-p."""
    # Apply temperature scaling
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Apply Top-K filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1)) # Ensure top_k is not larger than vocab size
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    # Apply Top-P (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (nucleus)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

    # Sample from the filtered distribution
    probabilities = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token

# --- Main Execution --- 

def main():
    parser = argparse.ArgumentParser(description="Run inference with the dual-head Qwen3 model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the fine-tuned model directory.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Input prompt for generation.")
    parser.add_argument("--output_wav", type=str, default=OUTPUT_FILENAME, help="Path to save the output audio file.")
    parser.add_argument("--max_new_text_tokens", type=int, default=MAX_NEW_TEXT_TOKENS, help="Maximum number of new text tokens to generate.")
    parser.add_argument("--codes_per_step", type=int, default=CODES_PER_STEP, help="Number of Mimi codes to generate per text token (heuristic).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling.")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 1. Load models
    try:
        model, tokenizer, mimi_decoder = load_models(args.model_path, DEVICE)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 2. Generate text and Mimi codes
    start_time = time.time()
    generated_text, generated_mimi_codes = generate_interleaved(
        model,
        tokenizer,
        args.prompt,
        max_new_text_tokens=args.max_new_text_tokens,
        codes_per_step=args.codes_per_step,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    end_time = time.time()
    print(f"\nGeneration took {end_time - start_time:.2f} seconds.")

    # 3. Decode Mimi codes to audio
    if generated_mimi_codes:
        decode_mimi_codes(mimi_decoder, generated_mimi_codes, MIMI_EXPECTED_SR, args.output_wav)
    else:
        print("No Mimi codes were generated, skipping audio decoding.")

if __name__ == "__main__":
    main() 