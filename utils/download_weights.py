"""
Utility for downloading pretrained weights.

Prefers direct .pth download from Hugging Face; falls back to Google Drive zip
if the requested filename is not available or download fails.
"""

import os
import zipfile
from pathlib import Path

# Direct .pth URL (no zip) - Tencent MedicalNet on Hugging Face
HF_MEDICALNET_REPO = "TencentMedicalNet/MedicalNet-Resnet18"
# Google Drive fallback (zip) file ID
DRIVE_FILE_ID = "13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG"


def _download_url_to_file(url: str, path: str) -> None:
    """Download from URL to file using urllib (no extra deps)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        with open(path, "wb") as out:
            out.write(response.read())


def _download_from_huggingface(output_dir: str, filename: str) -> str:
    """
    Download a single .pth file from Hugging Face (direct file, no zip).
    Returns path to the file, or raises on failure.
    """
    # Only resnet_18_23dataset.pth (and resnet_18.pth) are available as direct files
    if filename not in ("resnet_18_23dataset.pth", "resnet_18.pth"):
        raise FileNotFoundError(f"Direct Hugging Face download only supports resnet_18_23dataset.pth / resnet_18.pth, got {filename}")

    url = f"https://huggingface.co/{HF_MEDICALNET_REPO}/resolve/main/{filename}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(output_dir, filename)
    print(f"Downloading MedicalNet weights (direct .pth) from Hugging Face...")
    _download_url_to_file(url, weights_path)
    print(f"Downloaded to {weights_path}")
    return weights_path


def _download_from_google_drive_zip(output_dir: str, filename: str) -> str:
    """
    Download MedicalNet weights as a zip from Google Drive, then extract the .pth.
    Used when direct Hugging Face download is not available or fails.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(output_dir, filename)
    zip_path = os.path.join(output_dir, "weights.zip")

    if os.path.exists(zip_path):
        print(f"Found existing zip at {zip_path}, extracting...")
    else:
        print("Downloading MedicalNet weights (zip) from Google Drive...")
        try:
            try:
                import gdown
                gdown.download(
                    f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
                    zip_path,
                    quiet=False,
                    proxy=None,
                )
            except ImportError:
                import urllib.request
                import ssl
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
                url = f"https://drive.usercontent.com/download?id={DRIVE_FILE_ID}&export=download"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
                urllib.request.install_opener(opener)
                with opener.open(req) as resp, open(zip_path, "wb") as f:
                    f.write(resp.read())
            print(f"Downloaded zip to {zip_path}")
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise e

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    nested = os.path.join(output_dir, "pretrain")
    if os.path.exists(nested):
        src = os.path.join(nested, filename)
        if os.path.exists(src):
            os.replace(src, weights_path)
    if not os.path.exists(weights_path):
        src = os.path.join(output_dir, filename)
        if os.path.exists(src):
            os.replace(src, weights_path)

    import shutil
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.abspath(entry_path) == os.path.abspath(weights_path):
            continue
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        except Exception as ex:
            print(f"Warning: could not remove {entry_path}: {ex}")

    if os.path.exists(zip_path):
        os.remove(zip_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found after extraction: {weights_path}")
    return weights_path


def download_medicalnet_weights(
    output_dir: str = "pretrained",
    filename: str = "resnet_18_23dataset.pth",
    prefer_direct: bool = True,
) -> str:
    """
    Download MedicalNet pretrained ResNet-18 weights.

    By default downloads only the .pth file from Hugging Face (no zip).
    If that fails or the requested filename is not available as a direct file,
    falls back to downloading the full zip from Google Drive.

    Args:
        output_dir: Directory to save the weights
        filename: Name of the weights file (e.g. resnet_18_23dataset.pth)
        prefer_direct: If True, try Hugging Face direct .pth first; else use zip only

    Returns:
        Path to the downloaded weights file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(output_dir, filename)

    if os.path.exists(weights_path):
        print(f"Weights already exist at {weights_path}")
        return weights_path

    if prefer_direct and filename in ("resnet_18_23dataset.pth", "resnet_18.pth"):
        try:
            return _download_from_huggingface(output_dir, filename)
        except Exception as e:
            print(f"Direct download failed ({e}), falling back to Google Drive zip...")

    return _download_from_google_drive_zip(output_dir, filename)


if __name__ == "__main__":
    # Test the download
    weights_file = download_medicalnet_weights()
    print(f"Weights ready at: {weights_file}")
