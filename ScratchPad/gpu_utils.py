import torch

def print_gpu_info():
    """Prints CUDA GPU information if available."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    gpu_id = torch.cuda.current_device()

    print(f"GPU : {torch.cuda.get_device_name(gpu_id)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(gpu_id)}")

    gpu_props = torch.cuda.get_device_properties(gpu_id)

    print(f"GPU Memory : {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"Multiprocessors: {gpu_props.multi_processor_count}")
    print(f"Max Threads/MP : {gpu_props.max_threads_per_multi_processor}")
