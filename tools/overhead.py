
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import time
import gc

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_gpu_memory(model: nn.Module, input_size: Tuple, device: str = 'cuda') -> Dict[str, float]:
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, skipping GPU memory measurement")
        return {}
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    baseline_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    model = model.to(device)
    model.eval()
    
    model_memory = torch.cuda.memory_allocated(device) / 1024**2
    model_size = model_memory - baseline_memory

    dummy_input = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    forward_memory = torch.cuda.memory_allocated(device) / 1024**2
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    return {
        'model_size_mb': model_size,
        'forward_pass_mb': forward_memory - baseline_memory,
        'peak_memory_mb': peak_memory - baseline_memory,
        'baseline_mb': baseline_memory
    }


def measure_gpu_memory_training(model: nn.Module, input_size: Tuple, device: str = 'cuda', 
                                 optimizer_type: str = 'adam') -> Dict[str, float]:
    if not torch.cuda.is_available():
        print("Warning: CUDA not available")
        return {}
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    baseline_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    model = model.to(device)
    model.train()
    
    model_memory = torch.cuda.memory_allocated(device) / 1024**2
    model_size = model_memory - baseline_memory

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    optimizer_memory = torch.cuda.memory_allocated(device) / 1024**2
    optimizer_size = optimizer_memory - model_memory

    dummy_input = torch.randn(*input_size).to(device)
    dummy_target = torch.randn(input_size[0], 10).to(device)  
    criterion = torch.nn.CrossEntropyLoss()
    
    output = model(dummy_input)
    [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, region_feats_proj=output
    loss = criterion(cls_score, dummy_target.argmax(dim=1))
    loss.backward()  
    optimizer.step()
    optimizer.zero_grad()
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    return {
        'model_size_mb': model_size,
        'optimizer_size_mb': optimizer_size,
        'peak_training_memory_mb': peak_memory - baseline_memory,
        'total_mb': peak_memory - baseline_memory
    }


def calculate_flops_manual(model: nn.Module, input_size: Tuple, device: str = 'cuda') -> int:
    try:
        from thop import profile, clever_format
        
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(*input_size).to(device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.20f")
        print(flops)
        exit()
        
        return {'flops': flops, 'params': params}
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        print("Attempting to use fvcore instead...")
        
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count
            
            model = model.to(device)
            model.eval()
            dummy_input = torch.randn(*input_size).to(device)
            
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()

            if total_flops >= 1e9:
                flops_str = f"{total_flops / 1e9:.3f} G"
            elif total_flops >= 1e6:
                flops_str = f"{total_flops / 1e6:.3f} M"
            else:
                flops_str = f"{total_flops:.3f}"
            
            return {'flops': flops_str, 'flops_raw': total_flops}
        except ImportError:
            print("Warning: fvcore not installed. Install with: pip install fvcore")
            return {'flops': 'N/A', 'flops_raw': 0}


def profile_model(model: nn.Module, 
                  input_size: Tuple = (1, 3, 256, 128),
                  device: str = 'cuda',
                  model_name: str = "Model") -> Dict:

    print("\n" + "="*80)
    print(f"Profiling {model_name}")
    print("="*80)
    
    results = {}
    
    total_params, trainable_params = count_parameters(model)
    results['total_parameters'] = total_params
    results['trainable_parameters'] = trainable_params
    
    print(f"\nParameter Count:")
    print(f"  - Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    print(f"\n FLOPs Calculation:")
    flops_info = calculate_flops_manual(model, input_size, device)
    results.update(flops_info)
    
    if isinstance(flops_info.get('flops'), str):
        print(f"  - FLOPs: {flops_info['flops']}")
    else:
        print(f"  - FLOPs: {flops_info.get('flops', 'N/A')}")
    
    if torch.cuda.is_available():
        print(f"\n GPU Memory Footprint:")
        memory_info = measure_gpu_memory_training(model, input_size, device)
        results.update(memory_info)
        
        #print(f"  - Model Size: {memory_info.get('model_size_mb', 0):.2f} MB")
        #print(f"  - Forward Pass: {memory_info.get('forward_pass_mb', 0):.2f} MB")
        print(f"  - Peak Memory: {memory_info.get('peak_training_memory_mb', 0):.2f} MB")
    else:
        print("\n GPU Memory Footprint: CUDA not available")
    
    print(f"\n  Inference Speed:")
    inference_time = measure_inference_speed(model, input_size, device, num_iterations=100)
    results['inference_time_ms'] = inference_time
    print(f"  - Average Inference Time: {inference_time:.2f} ms")
    
    print("="*80 + "\n")
    
    return results


def measure_inference_speed(model: nn.Module, 
                           input_size: Tuple,
                           device: str = 'cuda',
                           num_iterations: int = 100,
                           warmup: int = 10) -> float:

    if not torch.cuda.is_available():
        device = 'cpu'
    
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  
    
    return avg_time


def save_profiling_results(results: Dict, save_path: str = "model_profiling_results.txt"):
    with open(save_path, 'w') as f:
        f.write("computational overhead\n")
        f.write("="*80 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    print("Model Profiler Tool")
    print("This is a utility module. Import and use profile_model() in your training scripts.")
    print("\nExample usage:")
    print("  from model_profiler import profile_model")
    print("  results = profile_model(model, input_size=(1, 3, 256, 128), model_name='MyModel')")
