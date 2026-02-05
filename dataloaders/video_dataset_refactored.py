import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def flow_func(X, of_skip=1, polar=False):
    # X is [T, H, W, C] (BGR or RGB)
    # Returns [T_of, H, W, 2]
    T, H, W, C = X.shape
    X_gray = [cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY) for x in X]
    
    # Output size depends on skip.
    # If we have T frames, and skip 1, we get T-1 flows.
    # The original code did: for j in range(0, X.shape[0]-of_skip, of_skip)
    # This implies a stride of `of_skip` and window `of_skip`.
    
    flows = []
    for j in range(0, T - of_skip, of_skip):
        prev = X_gray[j]
        curr = X_gray[j + of_skip]
        f = cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f = cv2.normalize(f, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        if polar:
            mag, ang = cv2.cartToPolar(f[:,:,0], f[:,:,1])
            f = np.concatenate([np.expand_dims(mag, axis=2), np.expand_dims(ang, axis=2)], axis=2)
        
        flows.append(f)
    
    if not flows:
        return np.zeros((0, H, W, 2), dtype=np.float32)

    return np.array(flows)


class VideoDatasetRefactored(Dataset):
    """
    Refactored VideoDataset with:
    1. Automated configuration (no hardcoded paths).
    2. Optimized loading (selective frame read).
    3. Integrated Optical Flow (optional).
    """

    def __init__(self, root_dir, output_dir, split='train', clip_len=16, 
                 preprocess=False, augment=True, compute_flow=False,
                 resize_height=128, resize_width=171, crop_size=112):
        
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.split = split
        self.clip_len = clip_len
        self.augment = augment
        self.compute_flow = compute_flow
        
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size

        if not self.check_integrity():
            raise RuntimeError(f'Dataset root path not found: {self.root_dir}')

        if (not self.check_preprocess()) or preprocess:
            print(f'Preprocessing dataset to {self.output_dir}...')
            self.preprocess()
        
        self.split_dir = os.path.join(self.output_dir, split)
        
        self.fnames, self.labels = [], []
        # Expecting structure: output_dir/split/class_label/video_name/frames...
        if not os.path.exists(self.split_dir):
             raise RuntimeError(f'Split directory not found: {self.split_dir}')

        classes = sorted(os.listdir(self.split_dir))
        # Filter out hidden files
        classes = [c for c in classes if not c.startswith('.')]
        
        self.label2index = {label: index for index, label in enumerate(classes)}
        
        for label in classes:
            class_path = os.path.join(self.split_dir, label)
            if not os.path.isdir(class_path):
                continue
            videos = sorted(os.listdir(class_path))
            for vid in videos:
                vid_path = os.path.join(class_path, vid)
                if os.path.isdir(vid_path):
                    self.fnames.append(vid_path)
                    self.labels.append(label)

        print(f'Number of {split} videos: {len(self.fnames)}')
        self.label_array = np.array([self.label2index[l] for l in self.labels], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Optimized loading: Only load frames we need.
        buffer = self.load_clip(self.fnames[index])
        
        # Buffer is [T, H, W, C]
        if self.augment:
            buffer = self.randomflip(buffer)
            
        # Compute Optical Flow if requested (BEFORE normalization)
        flow_buffer = None
        if self.compute_flow:
            # Buffer is [T, H, W, C], values 0-255 (float)
            # Compute flow.
            # Original code logic: T=16, returns T=16?
            # Original code: X_of = np.zeros([int(X.shape[0]/of_skip), ...])
            # Loop runs 15 times for T=16. The last frame is 0.
            
            # We use of_skip=1 by default for now.
            T, H, W, C = buffer.shape
            flow_buffer = np.zeros((T, H, W, 2), dtype=np.float32)
            
            for i in range(T - 1):
                prev = buffer[i].astype(np.uint8)
                curr = buffer[i+1].astype(np.uint8)
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                
                f = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Normalize to 0-1 range? Original code uses cv2.normalize with MINMAX to (0,1)?
                # Original: cv2.normalize(..., alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, ...)
                f = cv2.normalize(f, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                flow_buffer[i] = f
            
            # flow_buffer is [T, H, W, 2]
            # Transpose to [C, T, H, W] -> [2, T, H, W]
            flow_buffer = flow_buffer.transpose((3, 0, 1, 2))

        buffer = self.normalize(buffer) # [T, H, W, C]
        buffer = self.to_tensor(buffer) # [C, T, H, W]
        
        if self.compute_flow:
            return torch.from_numpy(buffer), torch.from_numpy(flow_buffer), self.label_array[index]
        else:
            return torch.from_numpy(buffer), self.label_array[index]

    def check_integrity(self):
        return os.path.exists(self.root_dir)

    def check_preprocess(self):
        return os.path.exists(self.output_dir) and os.path.exists(os.path.join(self.output_dir, 'train'))

    def preprocess(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for s in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, s), exist_ok=True)

        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            if os.path.isdir(file_path):
                video_files = [n for n in os.listdir(file_path) if not n.startswith('.')]
                if not video_files: continue
                
                # Simple split (can be improved)
                if len(video_files) < 2:
                    train = video_files
                    val = []
                    test = []
                else:
                    try:
                        train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                        if len(train_and_valid) < 2:
                            train = train_and_valid
                            val = []
                        else:
                            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
                    except ValueError:
                         # Fallback for edge cases
                         train = video_files
                         val = []
                         test = []
                
                # Process splits
                all_splits = []
                for v in train: all_splits.append((v, 'train'))
                for v in val: all_splits.append((v, 'val'))
                for v in test: all_splits.append((v, 'test'))

                for vid, s in all_splits:
                    dest_dir = os.path.join(self.output_dir, s, file)
                    self.process_video(vid, file, dest_dir)

    def process_video(self, video, action_name, save_dir):
        # Same logic as original but generalized
        video_filename = video.split('.')[0]
        target_dir = os.path.join(save_dir, video_filename)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        while count < frame_count:
            ret, frame = capture.read()
            if not ret: break
            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(os.path.join(target_dir, '0000{}.jpg'.format(i)), frame)
                i += 1
            count += 1
        capture.release()

    def load_clip(self, file_dir):
        # Optimization: List all frames, select indices, load ONLY those.
        all_frames = sorted(glob.glob(os.path.join(file_dir, '*.jpg')))
        frame_count = len(all_frames)
        
        # Logic from crop/center_crop moved here to determine indices
        if frame_count < self.clip_len:
            # Loop video if too short? Or duplicate?
            # For now, just take what we have and repeat last frame?
            # Or assume preprocessing ensured >= 16 frames (it tries to).
            indices = np.arange(frame_count)
            # Pad if necessary (simple padding)
            while len(indices) < self.clip_len:
                indices = np.concatenate([indices, [indices[-1]]])
            indices = indices[:self.clip_len]
            start_index = 0 
        else:
            if self.split == 'train': # temporal jitter
                start_index = np.random.randint(frame_count - self.clip_len)
            else: # center crop time
                start_index = max(0, (frame_count - self.clip_len) // 2)
            indices = np.arange(start_index, start_index + self.clip_len)
            
        # Select frame paths
        selected_frames = [all_frames[i] for i in indices]
        
        # Load images
        buffer = []
        for p in selected_frames:
            img = cv2.imread(p)
            if img is None:
                # Fallback or error
                continue
            buffer.append(img.astype(np.float64))
        
        buffer = np.array(buffer) # [T, H, W, C]

        # Now Perform Spatial Crop
        # Spatial crop logic
        if self.augment and self.split == 'train':
            height_index = np.random.randint(buffer.shape[1] - self.crop_size)
            width_index = np.random.randint(buffer.shape[2] - self.crop_size)
        else:
            height_index = max(0, (buffer.shape[1] - self.crop_size) // 2)
            width_index = max(0, (buffer.shape[2] - self.crop_size) // 2)

        buffer = buffer[:, height_index:height_index + self.crop_size,
                        width_index:width_index + self.crop_size, :]
        
        return buffer

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            # Buffer is [T, H, W, C]
            # cv2.flip(img, 1) flips horizontal
            new_buffer = []
            for frame in buffer:
                new_buffer.append(cv2.flip(frame, 1)) # Note: cv2.flip returns a copy
            buffer = np.array(new_buffer)
            # Make sure it's 4D
            if len(buffer.shape) == 3:
                buffer = np.expand_dims(buffer, axis=0)
        return buffer

    def normalize(self, buffer):
        # Buffer [T, H, W, C]
        # Subtract mean
        buffer -= np.array([[[90.0, 98.0, 102.0]]])
        return buffer

    def to_tensor(self, buffer):
        # [T, H, W, C] -> [C, T, H, W]
        return buffer.transpose((3, 0, 1, 2))

