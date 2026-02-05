import os
import sys
import subprocess
import urllib.request
from torchvision.datasets.utils import download_and_extract_archive, download_url

def download_ucf101(dest_dir='./ucf101'):
    """
    Downloads and extracts the UCF101 dataset and its official splits.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    # Official UCF101 URLs
    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    split_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    # 1. Download and Extract Splits (Zip is easy)
    print("Downloading Train/Test splits...")
    splits_zip = os.path.join(dest_dir, "splits.zip")
    if not os.path.exists(os.path.join(dest_dir, "ucfTrainTestlist")):
        download_and_extract_archive(split_url, download_root=dest_dir, filename="splits.zip", remove_finished=True)
        print("Splits extracted.")
    else:
        print("Splits already exist.")

    # 2. Download and Extract Videos (RAR is tricky)
    print("Downloading UCF101 Videos (approx 6.5GB)...")
    rar_path = os.path.join(dest_dir, "UCF101.rar")
    
    if not os.path.exists(os.path.join(dest_dir, "UCF-101")) and not os.path.exists(os.path.join(dest_dir, "ApplyEyeMakeup")):
        # Check if rar exists, if not download
        if not os.path.exists(rar_path):
            try:
                download_url(video_url, root=dest_dir, filename="UCF101.rar")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Trying alternative mirror...")
                # Mirror if needed: mirror_url = "..."
                return

        print("Extracting Videos. This requires unrar or 7z to be installed...")
        
        # Try 7z (common on many systems)
        try:
            subprocess.run(["7z", "x", rar_path, f"-o{dest_dir}"], check=True)
            print("Extraction with 7z successful.")
            os.remove(rar_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try unrar
                subprocess.run(["unrar", "x", rar_path, dest_dir], check=True)
                print("Extraction with unrar successful.")
                os.remove(rar_path)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("FAILED: Could not find 7z or unrar. Please install one of them.")
                print(f"The file is located at: {rar_path}")
                print("Manual extraction required: extract it into the directory so that categories are immediate subdirs.")

    else:
        print("Videos already appear to be extracted.")

    # Move files if they are nested inside an extra 'UCF101' folder
    nested_path = os.path.join(dest_dir, "UCF101")
    if os.path.exists(nested_path):
        print("Flattening directory structure...")
        for category in os.listdir(nested_path):
            src = os.path.join(nested_path, category)
            dst = os.path.join(dest_dir, category)
            if not os.path.exists(dst):
                os.rename(src, dst)
        os.rmdir(nested_path)
        print("Structure flattened.")

if __name__ == "__main__":
    # Allow overriding path via CLI
    path = sys.argv[1] if len(sys.argv) > 1 else './ucf101'
    download_ucf101(path)
