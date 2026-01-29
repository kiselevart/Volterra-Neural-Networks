import argparse
import gc
import os
import sys
import time
import torch

# Ensure repo root is on sys.path when running as a standalone script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.model_factory import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Find max batch size that fits in VRAM")
    parser.add_argument("--task", type=str, required=True, choices=["cifar", "video"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "ucf10", "ucf101", "hmdb51"])
    parser.add_argument("--model", type=str, required=True, choices=["vnn_simple", "vnn_ortho", "resnet18", "vnn_rgb", "vnn_fusion"])
    parser.add_argument("--Q", type=int, default=2, help="Volterra interaction factor (for VNNs)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batch", type=int, default=2048, help="Upper bound to try")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--timeout_s", type=float, default=20.0, help="Timeout (seconds) per batch attempt")
    parser.add_argument("--verbose", action="store_true", help="Print detailed probe logs")
    parser.add_argument("--confirm_batch", type=int, default=None, help="Confirm a specific batch size with repeated runs")
    parser.add_argument("--confirm_iters", type=int, default=3, help="Iterations for confirm_batch")
    return parser.parse_args()


def make_dummy_input(args, batch_size, device, dtype):
    if args.task == "cifar":
        x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        return x
    # video: assume 3x16x112x112 clips; flow stream has 2 channels
    if args.model == "vnn_fusion":
        rgb = torch.randn(batch_size, 3, 16, 112, 112, device=device, dtype=dtype)
        flow = torch.randn(batch_size, 2, 16, 112, 112, device=device, dtype=dtype)
        return [rgb, flow]
    return torch.randn(batch_size, 3, 16, 112, 112, device=device, dtype=dtype)


def try_batch(model, args, batch_size, device, dtype, attempt, total_attempts):
    torch.cuda.empty_cache()
    gc.collect()
    use_autocast = dtype == torch.float16
    input_dtype = torch.float32 if use_autocast else dtype
    try:
        with torch.no_grad():
            if args.verbose:
                print(f"[Attempt {attempt}/{total_attempts}] Trying batch_size={batch_size}")
            x = make_dummy_input(args, batch_size, device, input_dtype)
            start = time.time()
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            if elapsed > args.timeout_s:
                if args.verbose:
                    print(f"[Attempt {attempt}/{total_attempts}] TIMEOUT batch_size={batch_size} ({elapsed:.2f}s)")
                return False
        if args.verbose:
            print(f"[Attempt {attempt}/{total_attempts}] OK batch_size={batch_size}")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if args.verbose:
                print(f"[Attempt {attempt}/{total_attempts}] OOM batch_size={batch_size}")
            return False
        raise


def confirm_batch_size(model, args, batch_size, device, dtype):
    torch.cuda.empty_cache()
    gc.collect()
    use_autocast = dtype == torch.float16
    input_dtype = torch.float32 if use_autocast else dtype
    torch.cuda.reset_peak_memory_stats()
    times = []

    with torch.no_grad():
        for i in range(args.confirm_iters):
            x = make_dummy_input(args, batch_size, device, input_dtype)
            start = time.time()
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
            if args.verbose:
                print(f"[Confirm {i+1}/{args.confirm_iters}] batch_size={batch_size} time={elapsed:.3f}s")

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_time = sum(times) / len(times) if times else 0.0
    print(f"Confirmed batch_size={batch_size} | avg_time={avg_time:.3f}s | peak_mem={peak_mem:.2f} GB")


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to probe VRAM limits.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Set num_classes from dataset
    if args.dataset in ["cifar10", "ucf10"]:
        args.num_classes = 10
    elif args.dataset == "ucf101":
        args.num_classes = 101
    elif args.dataset == "hmdb51":
        args.num_classes = 51

    device = torch.device("cuda")
    model = get_model(args, device)
    model.eval()

    if args.confirm_batch is not None:
        confirm_batch_size(model, args, args.confirm_batch, device, dtype)
        return

    low = 1
    high = args.max_batch
    best = 0
    total_attempts = 0
    tmp_low, tmp_high = low, high
    while tmp_low <= tmp_high:
        total_attempts += 1
        mid = (tmp_low + tmp_high) // 2
        tmp_low = mid + 1

    attempt = 0
    while low <= high:
        attempt += 1
        mid = (low + high) // 2
        if args.verbose:
            print(f"[Attempt {attempt}/{total_attempts}] range=({low},{high}) -> mid={mid}")
        ok = try_batch(model, args, mid, device, dtype, attempt, total_attempts)
        if ok:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"Max batch size (approx): {best}")


if __name__ == "__main__":
    main()
