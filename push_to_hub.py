from huggingface_hub import HfApi, create_repo
import os

def push_to_huggingface_hub():
    """
    Pushes the contents of a local directory to a Hugging Face Hub repository.
    """
    local_folder_path = "qwen3-0.6B-dual-head-full-run1/checkpoint-1000"
    repo_id = "Mazino0/qwen2head-1000"
    
    # Ensure the local folder exists
    if not os.path.isdir(local_folder_path):
        print(f"Error: Local folder '{local_folder_path}' not found.")
        print("Please make sure the script is in the parent directory of 'qwen3-0.6B-dual-head-full-run1'")
        return

    print(f"Starting upload of '{local_folder_path}' to '{repo_id}'...")

    try:
        api = HfApi()

        # Create the repository if it doesn't exist
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository '{repo_id}' ensured to exist.")

        api.upload_folder(
            folder_path=local_folder_path,
            repo_id=repo_id,
            repo_type="model",  # Can be "model", "dataset", or "space"
            commit_message="Upload checkpoint-1000 files"
        )
        print(f"Successfully uploaded files to '{repo_id}'.")
        print(f"You can view your repository at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        print("Please ensure you have 'huggingface_hub' installed and are logged in (huggingface-cli login).")

if __name__ == "__main__":
    # Check if the script is in the correct directory relative to the files
    # The script expects to be in 'qwen2head', and the files in 'qwen2head/qwen3-0.6B-dual-head-full-run1/checkpoint-1000'
    expected_checkpoint_dir = "qwen3-0.6B-dual-head-full-run1/checkpoint-1000"
    if not os.path.exists(expected_checkpoint_dir):
        print(f"Error: The directory '{expected_checkpoint_dir}' was not found in the current location.")
        print(f"Please run this script from the 'qwen2head' directory, which should contain '{expected_checkpoint_dir}'.")
        print(f"Current working directory: {os.getcwd()}")
    else:
        push_to_huggingface_hub() 