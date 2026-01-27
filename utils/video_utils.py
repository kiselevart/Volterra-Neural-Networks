import numpy as np
import cv2
import torch

def calculate_video_flow(video_tensor, of_skip=1, polar=False):
    """
    Computes optical flow for a single video tensor.
    Input: video_tensor (C, T, H, W) normalized or uint8
    Output: flow_tensor (2, T//skip, H, W)
    """
    # Convert to (T, H, W, C) and numpy
    vid = video_tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
    
    # Ensure uint8 for OpenCV
    if vid.dtype != np.uint8:
        if vid.max() <= 1.0:
            vid = (vid * 255).astype(np.uint8)
        else:
            vid = vid.astype(np.uint8)

    T, H, W, C = vid.shape
    # Gray scale conversion
    frames_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in vid]
    
    flows = []
    
    for i in range(0, T - of_skip, of_skip):
        prev = frames_gray[i]
        next_ = frames_gray[i + of_skip]
        
        # Farneback Optical Flow (Optimized parameters)
        flow = cv2.calcOpticalFlowFarneback(
            prev, next_, None, 0.5, 3, 12, 1, 5, 1.2, 0
        )
        
        # Normalize flow
        flow = cv2.normalize(flow, None, alpha=0, beta=1, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if polar:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = np.stack([mag, ang], axis=-1)
            
        flows.append(flow)

    if len(flows) == 0:
        return torch.zeros(2, 1, H, W).float()

    # Pad to match original T
    while len(flows) < T:
        flows.append(flows[-1])

    # Stack to (T_new, H, W, 2)
    flow_stack = np.stack(flows, axis=0)
    
    # Convert back to torch (2, T_new, H, W) -> (C, T, H, W)
    flow_tensor = torch.from_numpy(flow_stack).permute(3, 0, 1, 2).float()
    return flow_tensor
