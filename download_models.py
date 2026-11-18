"""Download the trained models (models.zip on Google Drive) and unzip them into models/.

Weights are not stored in the Git repository (they are too large).
"""
import zipfile

import config


def main():
    if not config.DRIVE_FILE_ID:
        raise SystemExit("DRIVE_FILE_ID is empty in config.py. Train the models instead: python train_all.py")
    import gdown
    out = config.ROOT / "models.zip"
    gdown.download(f"https://drive.google.com/uc?id={config.DRIVE_FILE_ID}", str(out), quiet=False)
    with zipfile.ZipFile(out) as z:
        z.extractall(config.ROOT)
    out.unlink()
    print(f"Models ready in {config.MODELS_DIR}. Start the app with:  python app.py")


if __name__ == "__main__":
    main()
