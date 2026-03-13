import os


class Path:
    """Dataset path resolver used by legacy and unified loaders.

    Resolution order per dataset:
    1) Explicit environment variables (for remote/server use)
    2) Repository-local defaults under ./data
    """

    @staticmethod
    def _repo_data_dir() -> str:
        # mypath.py lives at repo root in this project.
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    @staticmethod
    def db_dir(database: str):
        db = (database or "").lower()
        data_root = Path._repo_data_dir()

        if db == "ucf101":
            root_dir = os.getenv("UCF101_ROOT", os.path.join(data_root, "ucf101"))
            output_dir = os.getenv("UCF101_PREPROCESSED", os.path.join(data_root, "ucf101_pre"))
        elif db == "hmdb51":
            root_dir = os.getenv("HMDB51_ROOT", os.path.join(data_root, "hmdb51"))
            output_dir = os.getenv("HMDB51_PREPROCESSED", os.path.join(data_root, "hmdb51_pre"))
        elif db == "ucf10":
            root_dir = os.getenv("UCF10_ROOT", os.path.join(data_root, "ucf10"))
            output_dir = os.getenv("UCF10_PREPROCESSED", os.path.join(data_root, "ucf10_pre"))
        elif db == "ucf11":
            root_dir = os.getenv("UCF11_ROOT", os.path.join(data_root, "ucf11"))
            output_dir = os.getenv("UCF11_PREPROCESSED", os.path.join(data_root, "ucf11_pre"))
        elif db == "cifar10":
            # Kept for compatibility; CIFAR loading currently uses torchvision root=./data.
            root_dir = os.getenv("CIFAR10_ROOT", os.path.join(data_root, "cifar-10-batches-py"))
            output_dir = os.getenv("CIFAR10_PREPROCESSED", os.path.join(data_root, "cifar10_pre"))
        else:
            raise NotImplementedError(f"Dataset not available: {database}")

        return os.path.abspath(root_dir), os.path.abspath(output_dir)
