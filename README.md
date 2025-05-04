# Qwen3 Dual-Head Text/Audio Code Experiment

This project experiments with modifying a Qwen3 language model (specifically configured via `config.py`) to have two output heads:

1.  **Text Head:** Generates standard text.
2.  **Mimi Code Head:** Generates audio codes using the first quantizer of the `kyutai/mimi` codec.

The goal is to train a model capable of simultaneously generating synchronized text and the corresponding audio representation.

## Project Structure

*   `config.py`: Central configuration file for model paths, data paths, training hyperparameters, loss weights, and Hugging Face Hub integration. **Modify this file first** to set up your experiment.
*   `prepare_data.py`: Script to process a source dataset (e.g., LibriTTS) by adding Mimi audio codes. Reads `config.py` for paths and settings. Uses a streaming approach with a generator to handle potentially large datasets.
*   `dual_head_model.py`: Defines the custom `Qwen3ForCausalLMWithMimiHead` model class, inheriting from `transformers.Qwen3ForCausalLM` and adding the second output head.
*   `train_dual_head.py`: Training script using the Hugging Face `Trainer`. Loads the custom model, tokenizer, processed data, and configuration from `config.py`. Handles dual-loss calculation and can push the final model to the Hugging Face Hub if configured.
*   `inference.py`: Script to load a trained dual-head model checkpoint and perform interleaved text/Mimi code generation (currently uses a heuristic approach).
*   `run_experiment.sh`: Simple bash script to sequentially run data preparation and training.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    *   Install `uv` (optional but recommended): `pip install uv`
    *   Install requirements:
        ```bash
        # Using uv
        uv pip install -r requirements.txt

        # Or using pip
        # pip install -r requirements.txt
        ```
    *   *(Note: A `requirements.txt` file should be created containing `torch`, `torchaudio`, `transformers`, `datasets`, `accelerate`, `sentencepiece`, `protobuf`, `huggingface_hub`, `moshi`)*

4.  **Configure:**
    *   Edit `config.py` to set:
        *   `model_config["model_name_or_path"]` (Base model like `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-4B`)
        *   `data_config["source_dataset_id"]` (e.g., `mythicinfinity/libritts`)
        *   `data_config["data_path"]` (Output path for processed data)
        *   `data_config["train_split"]`, `data_config["eval_split"]`
        *   `training_config["output_dir"]` (Where checkpoints are saved)
        *   Review other training hyperparameters (`batch_size`, `learning_rate`, etc.).
        *   Configure `training_config["push_to_hub"]`, `training_config["hub_model_id"]`, etc. for Hugging Face Hub integration.

## Workflow

1.  **Data Preparation:**
    *   Ensure `config.py` points to the correct source dataset and desired output path (`data_path`).
    *   Run the preparation script:
        ```bash
        python prepare_data.py
        ```
    *   This will load the source dataset splits specified in `config.py` using streaming, process them to add Mimi codes, and save the results to the directory specified in `data_config["data_path"]`. This can take a long time for large datasets.

2.  **Training:**
    *   **(Optional) Login to Hugging Face Hub:** If `push_to_hub` is `True` in `config.py`, log in first:
        ```bash
        huggingface-cli login
        ```
    *   Ensure `config.py` points to the *processed* data path (`data_config["data_path"]`) and specifies the correct training parameters and output directory.
    *   Run the training script directly:
        ```bash
        python train_dual_head.py
        ```
    *   Or use the experiment script:
        ```bash
        chmod +x run_experiment.sh
        ./run_experiment.sh
        ```
    *   Training logs and checkpoints will be saved to `training_config["output_dir"]`. If configured, the final model will be pushed to the Hugging Face Hub repository specified in `hub_model_id`.

3.  **Inference:**
    *   Once training is complete, use the inference script:
        ```bash
        python inference.py --model_path <path_to_your_checkpoint_dir> --prompt "Enter your text prompt here."
        ```
    *   Replace `<path_to_your_checkpoint_dir>` with the actual path (e.g., `./qwen3-4B-dual-head-full-run1`).
    *   This will generate text and corresponding Mimi codes (using the current heuristic strategy) and save the decoded audio to `output_audio.wav`.
