import os
import requests

FIGHT_MODEL_PATH = "models/fight_model.pt"


TRASH_MODEL_URL = (
    "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt"
)
TRASH_MODEL_PATH = "models/trash_model.pt"


def download_trash_model() -> str:
    """Download trash detection model if not already present. Returns local path."""
    if os.path.exists(TRASH_MODEL_PATH):
        return TRASH_MODEL_PATH

    os.makedirs("models", exist_ok=True)
    print(f"Downloading trash detection model to {TRASH_MODEL_PATH} ...")

    try:
        response = requests.get(TRASH_MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(TRASH_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Trash model downloaded successfully.")
        return TRASH_MODEL_PATH
    except Exception as e:
        print(f"Warning: Could not download trash model ({e}). Falling back to COCO proxies.")
        return ""


def download_fight_model() -> str:
    """Download fight detection model from HuggingFace if not already present."""
    if os.path.exists(FIGHT_MODEL_PATH):
        return FIGHT_MODEL_PATH

    os.makedirs("models", exist_ok=True)
    print("Downloading fight detection model from HuggingFace...")

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Musawer14/fight_detection_yolov8",
            filename="best.pt",
            local_dir="models",
        )
        os.rename(path, FIGHT_MODEL_PATH)
        print(f"Fight model saved to {FIGHT_MODEL_PATH}")
        return FIGHT_MODEL_PATH
    except Exception as e:
        print(f"Warning: Could not download fight model ({e}).")
        return ""
