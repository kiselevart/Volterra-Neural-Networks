import os
import urllib.request
import pathlib
import torch
import numpy as np
from network.fusion import vnn_rgb_of_highQ, vnn_fusion_highQ
import cv2
from joblib import Parallel, delayed

# Enable CuDNN autotune if available
torch.backends.cudnn.benchmark = True

# Public sample clip (tiny) from UCF101 mirror for demo purposes
SAMPLE_VIDEO_URL = (
    "https://github.com/academictorrents/volleyball/raw/master/UCF101_sample/v_TennisSwing_g02_c06.avi"
)
DEFAULT_VIDEO_DIR = pathlib.Path("./data/downloads")
DEFAULT_VIDEO_NAME = "v_TennisSwing_g02_c06.avi"
DEFAULT_VIDEO_PATH = DEFAULT_VIDEO_DIR / DEFAULT_VIDEO_NAME


def download_video(url: str = SAMPLE_VIDEO_URL, dest: pathlib.Path = DEFAULT_VIDEO_PATH) -> pathlib.Path:
    """Download a sample video if it is not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading sample video to {dest} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download video from {url}: {exc}") from exc
    return dest


def flow(X, Ht, Wd, of_skip=1, polar=False):
    X_of = np.zeros([int(X.shape[0] / of_skip), Ht, Wd, 2])
    of_ctr = -1
    for j in range(0, X.shape[0] - of_skip, of_skip):
        of_ctr += 1
        flow_map = cv2.normalize(
            cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(np.array(X[j + of_skip, :, :, :], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(np.array(X[j, :, :, :], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            ),
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        if polar:
            mag, ang = cv2.cartToPolar(flow_map[:, :, 0], flow_map[:, :, 1])
            X_of[of_ctr, :, :, :] = np.concatenate(
                [np.expand_dims(mag, axis=2), np.expand_dims(ang, axis=2)], axis=2
            )
        else:
            X_of[of_ctr, :, :, :] = flow_map
    return X_of


def compute_optical_flow(X, Ht, Wd, num_proc=4, of_skip=1, polar=False):
    X = (X.permute(0, 2, 3, 4, 1)).detach().cpu().numpy()
    optical_flow = Parallel(n_jobs=num_proc)(
        delayed(flow)(X[i], Ht, Wd, of_skip, polar) for i in range(X.shape[0])
    )
    X_of = torch.tensor(np.asarray(optical_flow)).float()
    return X_of.permute(0, 4, 1, 2, 3)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def load_class_names(path: str = "./dataloaders/ucf_labels.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    class_names = load_class_names()

    # Init models
    model_RGB = vnn_rgb_of_highQ.VNN(num_classes=101)
    model_OF = vnn_rgb_of_highQ.VNN(num_classes=101, num_ch=2)
    model_fuse = vnn_fusion_highQ.VNN_F(num_classes=101, num_ch=192)

    # Expect checkpoint path from env or default; raise if missing
    ckpt_path = os.environ.get(
        "VNN_FUSION_CHECKPOINT",
        "./models/VNN_Fusion-ucf101_epoch-99.pth.tar",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Set VNN_FUSION_CHECKPOINT to a valid path."
        )

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model_RGB.load_state_dict(checkpoint["state_dict_rgb"])
    model_OF.load_state_dict(checkpoint["state_dict_of"])
    model_fuse.load_state_dict(checkpoint["state_dict_fuse"])
    model_RGB.to(device)
    model_OF.to(device)
    model_fuse.to(device)

    model_RGB.eval()
    model_OF.eval()
    model_fuse.eval()

    # Download or reuse a sample video
    video_path = download_video()
    cap = cv2.VideoCapture(str(video_path))
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining or frame is None:
            break
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs_of = compute_optical_flow(inputs, 112, 112)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            inputs_of = torch.autograd.Variable(inputs_of, requires_grad=False).to(device)
            with torch.no_grad():
                outputs_rgb = model_RGB(inputs)
                outputs_of = model_OF(inputs_of)
                outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            clip = []

            cv2.putText(
                frame,
                class_names[label].split(" ")[-1].strip(),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                frame,
                f"prob: {probs[0][label]:.4f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

        cv2.imshow("result", frame)
        cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
