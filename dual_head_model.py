#!/usr/bin/env python
# coding: utf-8

"""Defines the Dual-Head Qwen model for text and Mimi code generation."""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union, Dict

# Import config explicitly to avoid name collision
import config as project_config

# Constants (now primarily from config)
# MIMI_CODEBOOK_SIZE = 2048 # Defined in config.py

class Qwen3ForCausalLMWithMimiHead(Qwen3ForCausalLM):
    """
    Qwen3 model with an additional linear head for predicting Mimi audio codes.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)

        # Use codebook size from project_config
        mimi_codebook_size = project_config.model_config.get("MIMI_CODEBOOK_SIZE", 2048) # Use project_config

        # Add the new head for Mimi code prediction
        # It takes the hidden states from the Qwen model and projects them to the Mimi codebook size.
        self.mimi_code_head = nn.Linear(config.hidden_size, mimi_codebook_size, bias=False)

        # Store loss weights from project_config for use in forward pass
        self.loss_weight_text = project_config.loss_config.get("loss_weight_text", 1.0)
        self.loss_weight_mimi = project_config.loss_config.get("loss_weight_mimi", 1.0)
        self.mimi_codebook_size_for_loss = mimi_codebook_size # Store for loss calculation

        # Initialize weights for the new head (optional but often recommended)
        # self.post_init() # Call post_init to handle weight initialization potentially
        # Or manually: self._init_weights(self.mimi_code_head)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # Text labels
        mimi_labels: Optional[torch.LongTensor] = None, # Mimi code labels (need to be handled in training)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Loss weights are now accessed via self
    ) -> Union[Tuple, CausalLMOutputWithPast, Dict]: # Return type might need custom class
        """
        Forward pass that computes logits for both text and Mimi codes.
        Adds `mimi_code_logits` to the output.
        Handles loss calculation if both `labels` (text) and `mimi_labels` are provided.
        Applies loss weighting based on config.training_config.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === 1. Get Base Qwen Model Outputs ===
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # Ensure we get hidden states
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # === 2. Calculate Text Logits ===
        lm_logits = self.lm_head(hidden_states)

        # === 3. Calculate Mimi Code Logits ===
        mimi_code_logits = self.mimi_code_head(hidden_states)

        # === 4. Calculate Losses (if labels provided) ===
        loss = None
        text_loss = None
        mimi_loss = None
        if labels is not None or mimi_labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = 0.0
            # Get weights and codebook size stored in self during init
            loss_weight_text = self.loss_weight_text
            loss_weight_mimi = self.loss_weight_mimi
            mimi_codebook_size = self.mimi_codebook_size_for_loss

            # --- Text Loss --- 
            if labels is not None:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                text_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                total_loss += loss_weight_text * text_loss # Apply weight

            # --- Mimi Code Loss --- 
            if mimi_labels is not None:
                shift_mimi_logits = mimi_code_logits[..., :-1, :].contiguous()
                shift_mimi_labels = mimi_labels[..., 1:].contiguous()
                mimi_loss = loss_fct(shift_mimi_logits.view(-1, mimi_codebook_size), shift_mimi_labels.view(-1))
                total_loss += loss_weight_mimi * mimi_loss # Apply weight

            loss = total_loss if (text_loss is not None or mimi_loss is not None) else None

        # === 5. Prepare Output ===
        if return_dict:
            output = {
                "loss": loss, 
                "text_loss": text_loss, 
                "mimi_loss": mimi_loss,
                "logits": lm_logits, 
                "mimi_code_logits": mimi_code_logits, 
                "past_key_values": transformer_outputs.past_key_values, 
                "hidden_states": transformer_outputs.hidden_states, 
                "attentions": transformer_outputs.attentions, 
            }
            return output
        else:
            # Note: Returning tuple might become complex to manage with added losses
            return (loss, lm_logits, mimi_code_logits) + transformer_outputs[1:]

# Example Usage (Conceptual - requires model loading and data)
if __name__ == '__main__':
    from transformers import AutoTokenizer

    # Use the correct Qwen3 identifier from config
    model_name = project_config.model_config["model_name_or_path"]
    print(f"Loading base model config: {model_name}")
    # Use Qwen3Config
    qwen_config = Qwen3Config.from_pretrained(model_name, trust_remote_code=True)

    print("Initializing custom dual-head model...")
    # Use Qwen3 class
    model = Qwen3ForCausalLMWithMimiHead(qwen_config)
    model.eval() # Set to eval mode for inference example

    print(f"Model Heads:")
    print(f"  LM Head (Text): {model.lm_head}")
    print(f"  Mimi Code Head: {model.mimi_code_head}")

    # Example dummy input
    print("\nCreating dummy input...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dummy_text = "Hello, world!"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Forward pass
    print("\nPerforming forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, return_dict=True)

    print("\nOutputs received:")
    print(f"  Keys: {outputs.keys()}")
    print(f"  Text Logits shape: {outputs['logits'].shape}")
    print(f"  Mimi Code Logits shape: {outputs['mimi_code_logits'].shape}")
    print(f"  Loss: {outputs.get('loss')}") # Will be None as no labels provided
    print(f"  Text Loss: {outputs.get('text_loss')}")
    print(f"  Mimi Loss: {outputs.get('mimi_loss')}")

    # Example loss calculation (dummy labels)
    print("\nTesting forward pass with dummy labels...")
    dummy_text_labels = inputs["input_ids"].clone()
    # Create dummy mimi labels (same shape as input_ids for this example)
    mimi_codebook_size = project_config.model_config.get("MIMI_CODEBOOK_SIZE", 2048) # Use project_config
    dummy_mimi_labels = torch.randint(0, mimi_codebook_size, dummy_text_labels.shape)

    with torch.no_grad(): # Still no_grad for testing print, use .train() and enable grad for actual training
        outputs_with_labels = model(
            input_ids=inputs["input_ids"],
            labels=dummy_text_labels,
            mimi_labels=dummy_mimi_labels,
            return_dict=True
        )

    print("\nOutputs with dummy labels:")
    print(f"  Keys: {outputs_with_labels.keys()}")
    print(f"  Loss: {outputs_with_labels.get('loss')}") # Should have a value now
    print(f"  Text Loss: {outputs_with_labels.get('text_loss')}") # Should have a value
    print(f"  Mimi Loss: {outputs_with_labels.get('mimi_loss')}") # Should have a value

    print("\nCustom model definition complete.") 