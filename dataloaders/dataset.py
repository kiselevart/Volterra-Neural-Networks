import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from mypath import Path
from utils.video_utils import calculate_video_flow


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, augment=True):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.augment = augment

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Per-channel BGR mean pixel values (C3D/Sports-1M convention).
        # Stored as an attribute so consumers (e.g. FlowDatasetWrapper) can
        # denormalize without hardcoding these values themselves.
        self.mean = np.array([90.0, 98.0, 102.0], dtype=np.float32)  # [B, G, R]

        # Detect whether root_dir has pre-split structure (train/val/test subdirs)
        # or flat class folders that need splitting
        self.pre_split = all(
            os.path.isdir(os.path.join(self.root_dir, s))
            for s in ('train', 'val', 'test')
        )

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.'
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        if self.augment:
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
        else:
            buffer = self.center_crop(buffer, self.clip_len, self.crop_size)
        buffer = self.ensure_clip_len(buffer, self.clip_len)
        labels = np.array(self.label_array[index])

        if self.augment:
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        """Check if preprocessed frames exist with correct dimensions."""
        if not os.path.exists(self.output_dir):
            return False
        train_dir = os.path.join(self.output_dir, 'train')
        if not os.path.exists(train_dir):
            return False

        for ii, video_class in enumerate(os.listdir(train_dir)):
            class_dir = os.path.join(train_dir, video_class)
            if not os.path.isdir(class_dir):
                continue
            for video in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, video)
                if not os.path.isdir(video_dir):
                    continue
                frames = sorted(os.listdir(video_dir))
                if not frames:
                    return False
                image = cv2.imread(os.path.join(video_dir, frames[0]))
                if image is None or image.shape[0] != 128 or image.shape[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)

        if self.pre_split:
            # Data already split into train/val/test with class subfolders of videos
            self._preprocess_pre_split()
        else:
            # Flat class folders — split ourselves
            self._preprocess_flat()

        print('Preprocessing finished.')

    def _is_video_file(self, path):
        return os.path.isfile(path) and path.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.mpeg', '.mov'))

    def _collect_video_entries(self, class_dir):
        """Collect supported videos in a class directory.

        Supports:
          1) Direct video files under class_dir
          2) One-level nested folders containing video files (e.g., UCF11 groups)

        Returns paths relative to class_dir suitable for joining later.
        """
        entries = []
        for name in sorted(os.listdir(class_dir)):
            full_path = os.path.join(class_dir, name)
            if self._is_video_file(full_path):
                entries.append(name)
                continue

            if os.path.isdir(full_path):
                for nested in sorted(os.listdir(full_path)):
                    nested_path = os.path.join(full_path, nested)
                    if self._is_video_file(nested_path):
                        entries.append(os.path.join(name, nested))

        return entries

    def _preprocess_pre_split(self):
        """Preprocess when root_dir already has train/val/test/class/video.avi structure."""
        for split in ['train', 'val', 'test']:
            split_src = os.path.join(self.root_dir, split)
            if not os.path.isdir(split_src):
                continue
            for action_name in sorted(os.listdir(split_src)):
                action_dir = os.path.join(split_src, action_name)
                if not os.path.isdir(action_dir):
                    continue
                save_dir = os.path.join(self.output_dir, split, action_name)
                os.makedirs(save_dir, exist_ok=True)
                for video in self._collect_video_entries(action_dir):
                    self.process_video(video, os.path.join(split, action_name), save_dir)

    def _preprocess_flat(self):
        """Preprocess when root_dir has flat class folders (original UCF101 layout)."""
        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            if os.path.isdir(file_path):
                video_files = self._collect_video_entries(file_path)
                if len(video_files) < 3:
                    continue

                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(self.output_dir, 'train', file)
                val_dir = os.path.join(self.output_dir, 'val', file)
                test_dir = os.path.join(self.output_dir, 'test', file)

                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                for video in train:
                    self.process_video(video, file, train_dir)

                for video in val:
                    self.process_video(video, file, val_dir)

                for video in test:
                    self.process_video(video, file, test_dir)

    def process_video(self, video, action_name, save_dir):
        """Extract and resize frames from a video file.
        
        Args:
            video: Video filename (e.g. 'v_Apply_g01_c01.avi')
            action_name: Relative path from root_dir to the folder containing the video
                         (e.g. 'ApplyEyeMakeup' or 'train/ApplyEyeMakeup')
            save_dir: Directory to save extracted frames
        """
        # Initialize a VideoCapture object to read video data into a numpy array
        src_path = os.path.join(self.root_dir, action_name, video)
        video_filename = os.path.splitext(os.path.basename(video))[0]
        target_dir = os.path.join(save_dir, video_filename)

        capture = cv2.VideoCapture(src_path)
        if not capture.isOpened():
            print(f"[WARN] Skipping unreadable video: {src_path}")
            capture.release()
            return

        os.makedirs(target_dir, exist_ok=True)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(target_dir, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

        if i == 0 and os.path.isdir(target_dir):
            try:
                os.rmdir(target_dir)
            except OSError:
                pass
        elif i > 0:
            self._compute_and_save_flow(target_dir)

    def _compute_and_save_flow(self, video_dir):
        """Compute optical flow from raw frames in video_dir and save to flow.npy.

        Operates on full-resolution frames (before any crop/normalize) so the
        stored flow can be spatially cropped at load time to match any RGB crop.
        Output: float32 array [2, T, H, W] saved as flow.npy alongside the frames.
        """
        frame_paths = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir) if f.endswith('.jpg')
        ])
        if len(frame_paths) < 2:
            return

        imgs = [cv2.imread(p) for p in frame_paths]
        imgs = [img for img in imgs if img is not None]
        if len(imgs) < 2:
            return

        # [T, H, W, C] → tensor [C, T, H, W], raw uint8 pixel values (no normalization)
        video_tensor = torch.from_numpy(
            np.stack(imgs, axis=0)
        ).permute(3, 0, 1, 2).float()

        flow = calculate_video_flow(video_tensor)  # [2, T, H, W]
        np.save(os.path.join(video_dir, "flow.npy"), flow.numpy())

    def ensure_flows(self):
        """Compute and cache flow.npy for any video directory that lacks one.

        Idempotent — skips videos that already have flow.npy. Intended for
        datasets preprocessed before flow caching was introduced.
        """
        missing = [
            d for d in self.fnames
            if not os.path.exists(os.path.join(d, "flow.npy"))
        ]
        if not missing:
            return
        print(f"==> Computing optical flow for {len(missing)} videos (one-time)...")
        for video_dir in tqdm(missing, desc="Flow"):
            self._compute_and_save_flow(video_dir)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i in range(len(buffer)):
                buffer[i] = cv2.flip(buffer[i], flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= self.mean
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir) if img.endswith('.jpg')])
        frame_count = len(frames)
        if frame_count == 0:
            return np.empty((0, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            if frame is None:
                frame = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.float32)
            else:
                frame = np.array(frame).astype(np.float64)
            buffer[i] = frame

        return buffer

    def ensure_clip_len(self, buffer, clip_len):
        """Guarantee temporal length equals clip_len for batching safety."""
        num_frames = buffer.shape[0]
        if num_frames == clip_len:
            return buffer

        if num_frames == 0:
            return np.zeros((clip_len, self.crop_size, self.crop_size, 3), dtype=np.float32)

        if num_frames > clip_len:
            return buffer[:clip_len]

        pad_count = clip_len - num_frames
        pad_frames = np.repeat(buffer[-1:, :, :, :], repeats=pad_count, axis=0)
        return np.concatenate([buffer, pad_frames], axis=0)

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        max_time = buffer.shape[0] - clip_len
        time_index = np.random.randint(max_time + 1) if max_time > 0 else 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def center_crop(self, buffer, clip_len, crop_size):
        # Center temporal clip
        time_index = max(0, (buffer.shape[0] - clip_len) // 2)
        height_index = max(0, (buffer.shape[1] - crop_size) // 2)
        width_index = max(0, (buffer.shape[2] - crop_size) // 2)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
