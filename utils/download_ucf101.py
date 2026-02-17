import os
import sys
import ssl
import subprocess
import zipfile

import certifi

# Fix macOS SSL certificate issue globally
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def _curl_download(url, dest_path):
    """Download a file using curl (ships with macOS), with progress bar."""
    print(f"  curl: {url}")
    subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", dest_path, url],
        check=True,
    )


def download_ucf101(dest_dir='./data/ucf101'):
    """
    Downloads and extracts the UCF101 dataset and its official splits.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Official UCF101 URLs
    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    split_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    # 1. Download and Extract Splits
    splits_dir = os.path.join(dest_dir, "ucfTrainTestlist")
    if not os.path.exists(splits_dir):
        print("Downloading Train/Test splits...")
        splits_zip = os.path.join(dest_dir, "splits.zip")
        _curl_download(split_url, splits_zip)
        with zipfile.ZipFile(splits_zip, 'r') as zf:
            zf.extractall(dest_dir)
        os.remove(splits_zip)
        print("Splits extracted.")
    else:
        print("Splits already exist.")

    # 2. Download and Extract Videos (RAR — requires unrar or 7z)
    rar_path = os.path.join(dest_dir, "UCF101.rar")
    has_videos = (
        os.path.exists(os.path.join(dest_dir, "UCF-101"))
        or os.path.exists(os.path.join(dest_dir, "ApplyEyeMakeup"))
    )

    if not has_videos:
        if not os.path.exists(rar_path):
            print("Downloading UCF101 Videos (approx 6.5GB)...")
            _curl_download(video_url, rar_path)

        print("Extracting Videos. This requires unrar or 7z to be installed...")
        extracted = False
        for tool, args in [
            ("7z", ["7z", "x", rar_path, f"-o{dest_dir}"]),
            ("unrar", ["unrar", "x", rar_path, dest_dir]),
        ]:
            try:
                subprocess.run(args, check=True)
                print(f"Extraction with {tool} successful.")
                os.remove(rar_path)
                extracted = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if not extracted:
            print("FAILED: Could not find 7z or unrar. Install one:")
            print("  brew install unrar   OR   brew install p7zip")
            print(f"  Then extract {rar_path} manually into {dest_dir}/")
            return
    else:
        print("Videos already appear to be extracted.")

    # Move files if they are nested inside an extra 'UCF-101' or 'UCF101' folder
    for nested_name in ("UCF-101", "UCF101"):
        nested_path = os.path.join(dest_dir, nested_name)
        if os.path.isdir(nested_path):
            print(f"Flattening {nested_name}/ into {dest_dir}/ ...")
            for category in os.listdir(nested_path):
                src = os.path.join(nested_path, category)
                dst = os.path.join(dest_dir, category)
                if os.path.isdir(src) and not os.path.exists(dst):
                    os.rename(src, dst)
            # Remove the now-empty nested dir (ignore if not empty)
            try:
                os.rmdir(nested_path)
            except OSError:
                pass
            print("Structure flattened.")
            break

if __name__ == "__main__":
    # Allow overriding path via CLI
    path = sys.argv[1] if len(sys.argv) > 1 else './data/ucf101'
    download_ucf101(path)
