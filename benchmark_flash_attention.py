#!/usr/bin/env python3
"""Performance benchmark script for Flash Attention vs Standard Attention."""

import sys
import os
import time
import torch
import json
import statistics
from datetime import datetime
from transformers import AutoTokenizer

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import ConfigManager
from model_manager import ModelManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise for benchmarking
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_flash_attention_compatibility():
    """Check if hardware supports Flash Attention efficiently."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    gpu_name = torch.cuda.get_device_name()
    compute_capability = torch.cuda.get_device_capability()
    
    # Flash Attention works best on A100, H100, RTX 30xx/40xx
    if compute_capability[0] < 7:  # Less than V100
        return False, f"GPU {gpu_name} may not benefit from Flash Attention"
    
    return True, f"GPU {gpu_name} (compute {compute_capability[0]}.{compute_capability[1]}) compatible"

def save_benchmark_results(results, filename=None):
    """Save benchmark results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def get_detailed_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    
    return (f"GPU Memory: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved\n"
            f"           {max_allocated:.3f}GB max allocated, {max_reserved:.3f}GB max reserved")

def reset_memory_stats():
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

def warmup_model(model_manager, num_warmup=3):
    """Perform warmup runs to stabilize model performance."""
    print("Running warmup...")
    warmup_prompts = ["Hello", "Test", "Warmup"]
    
    for i in range(num_warmup):
        prompt = warmup_prompts[i % len(warmup_prompts)]
        try:
            model_manager.generate_response(prompt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Warmup run {i+1} failed: {e}")

def count_tokens_accurate(text, model_name):
    """Accurately count tokens using the model's tokenizer."""
    try:
        # Try to load the tokenizer for accurate counting
        tokenizer = AutoTokenizer.from_pretrained(model_name.replace("-8bit", "").replace("-4bit", ""))
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text.split()) * 1.3

def calculate_statistics(values):
    """Calculate statistics for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values)
    }


def benchmark_attention(use_flash_attention: bool, num_tests: int = 3) -> dict:
    """Benchmark model performance with or without Flash Attention.
    
    Args:
        use_flash_attention: Whether to use Flash Attention
        num_tests: Number of test runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*50}")
    print(f"Benchmarking {'Flash Attention' if use_flash_attention else 'Standard Attention'}")
    print(f"{'='*50}")
    
    config_manager = ConfigManager()
    model_manager = ModelManager(config_manager)
    
    # Temporarily modify config to control Flash Attention
    model_name = "codellama-7b-instruct"
    model_settings = config_manager.get_settings(model_name)
    original_flash_setting = model_settings.get('use_flash_attention', False)
    
    # Override Flash Attention setting
    model_settings['use_flash_attention'] = use_flash_attention
    
    try:
        # Load model
        print("Loading model...")
        start_time = time.time()
        success = model_manager.load_model(model_name)
        load_time = time.time() - start_time
        
        if not success:
            raise RuntimeError("Failed to load model")
        
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Show memory after model loading
        memory_after_load = get_gpu_memory_usage()
        print(f"Memory after model loading: {memory_after_load['allocated']:.2f}GB allocated, {memory_after_load['reserved']:.2f}GB reserved")
        print(f"Detailed memory state:")
        print(get_detailed_memory_info())
        
        # Reset memory statistics for accurate measurement
        reset_memory_stats()
        
        # Perform warmup
        warmup_model(model_manager, num_warmup=1)  # Reduced from 2 to 1 for faster testing
        
        # Test prompts of varying lengths
        test_prompts = [
            "Hello!",
            "Write a short story about a robot learning to cook.",
            "Explain the concept of machine learning and its applications in modern technology. Include examples of how it's used in different industries and discuss both benefits and potential challenges.",
        ]
        
        results = {
            'flash_attention': use_flash_attention,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'load_time': load_time,
            'prompt_results': [],
            'summary_stats': {}
        }
        
        all_generation_times = []
        all_tokens_per_second = []
        all_memory_deltas = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTesting prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            prompt_times = []
            prompt_tps = []
            prompt_memory_deltas = []
            
            for run in range(num_tests):
                # Only clear cache before first run to see memory accumulation
                if run == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset peak memory stats for this measurement
                    torch.cuda.reset_peak_memory_stats()
                
                # Measure memory before generation (both allocated and reserved)
                memory_before = get_gpu_memory_usage()
                
                # Generate response
                start_time = time.time()
                response = model_manager.generate_response(prompt)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Measure memory after generation and peak usage
                memory_after = get_gpu_memory_usage()
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                
                # Calculate memory delta for both allocated and reserved
                memory_delta = {
                    'allocated': memory_after['allocated'] - memory_before['allocated'],
                    'reserved': memory_after['reserved'] - memory_before['reserved'],
                    'peak_allocated': peak_allocated - memory_before['allocated'],
                    'peak_reserved': peak_reserved - memory_before['reserved']
                }
                
                # Use peak memory change as the primary metric
                memory_delta_value = max(
                    abs(memory_delta['peak_allocated']), 
                    abs(memory_delta['peak_reserved']),
                    abs(memory_delta['allocated']),
                    abs(memory_delta['reserved'])
                )
                
                # Count tokens accurately
                response_tokens = count_tokens_accurate(response, model_name)
                tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0
                
                prompt_times.append(generation_time)
                prompt_tps.append(tokens_per_second)
                prompt_memory_deltas.append(memory_delta_value)
                
                print(f"  Run {run+1}: {generation_time:.2f}s, {tokens_per_second:.1f} tokens/s")
                print(f"    Memory delta: alloc={memory_delta['allocated']:.3f}GB, reserved={memory_delta['reserved']:.3f}GB")
                print(f"    Peak delta: alloc={memory_delta['peak_allocated']:.3f}GB, reserved={memory_delta['peak_reserved']:.3f}GB")
            
            # Calculate statistics for this prompt
            time_stats = calculate_statistics(prompt_times)
            tps_stats = calculate_statistics(prompt_tps)
            memory_stats = calculate_statistics(prompt_memory_deltas)
            
            prompt_result = {
                'prompt_index': i,
                'prompt_preview': prompt[:50],
                'prompt_length': len(prompt),
                'generation_time': time_stats,
                'tokens_per_second': tps_stats,
                'memory_delta': memory_stats,
                'num_runs': num_tests
            }
            
            results['prompt_results'].append(prompt_result)
            
            # Collect for overall statistics
            all_generation_times.extend(prompt_times)
            all_tokens_per_second.extend(prompt_tps)
            all_memory_deltas.extend(prompt_memory_deltas)
            
            print(f"  Average: {time_stats['mean']:.2f}±{time_stats['std']:.2f}s, {tps_stats['mean']:.1f}±{tps_stats['std']:.1f} tokens/s")
        
        # Calculate overall statistics
        results['summary_stats'] = {
            'generation_time': calculate_statistics(all_generation_times),
            'tokens_per_second': calculate_statistics(all_tokens_per_second),
            'memory_delta': calculate_statistics(all_memory_deltas)
        }
        
        print(f"\nOverall Summary:")
        print(f"  Generation Time: {results['summary_stats']['generation_time']['mean']:.2f}±{results['summary_stats']['generation_time']['std']:.2f}s")
        print(f"  Tokens/Second: {results['summary_stats']['tokens_per_second']['mean']:.1f}±{results['summary_stats']['tokens_per_second']['std']:.1f}")
        print(f"  Memory Delta: {results['summary_stats']['memory_delta']['mean']:.3f}±{results['summary_stats']['memory_delta']['std']:.3f}GB")
        
        # Unload model
        model_manager.unload_model()
        
        return results
        
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return None
    finally:
        # Restore original setting
        model_settings['use_flash_attention'] = original_flash_setting

def main():
    """Run performance comparison between Flash Attention and Standard Attention."""
    print("Flash Attention Performance Benchmark")
    print("=" * 60)
    
    # Check hardware compatibility
    compatible, message = check_flash_attention_compatibility()
    print(f"Hardware Compatibility: {message}")
    if not compatible:
        print("⚠️  Flash Attention may not provide benefits on this hardware")
    
    # Check if Flash Attention is available
    try:
        import flash_attn
        print(f"Flash Attention version: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention not available. Please install flash-attn.")
        return
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        initial_memory = get_gpu_memory_usage()
        print(f"Initial GPU Memory (before model loading): {initial_memory['allocated']:.2f}GB allocated, {initial_memory['reserved']:.2f}GB reserved")
    else:
        print("CUDA not available")
    
    # Benchmark both configurations
    num_tests = 2  # Reduced for faster testing - can be increased for more thorough results
    
    print(f"\nRunning {num_tests} test runs per prompt for statistical accuracy...")
    
    # Test standard attention
    print("\n" + "="*60)
    print("PHASE 1: Standard Attention Benchmark")
    print("="*60)
    standard_results = benchmark_attention(use_flash_attention=False, num_tests=num_tests)
    
    # Test Flash Attention
    print("\n" + "="*60)
    print("PHASE 2: Flash Attention Benchmark") 
    print("="*60)
    flash_results = benchmark_attention(use_flash_attention=True, num_tests=num_tests)
    
    # Compare results
    if standard_results and flash_results:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        # Load time comparison
        print(f"Load Time:")
        print(f"  Standard: {standard_results['load_time']:.2f}s")
        print(f"  Flash:    {flash_results['load_time']:.2f}s")
        load_speedup = standard_results['load_time']/flash_results['load_time'] if flash_results['load_time'] > 0 else 0
        print(f"  Speedup:  {load_speedup:.2f}x")
        
        # Generation speed comparison
        std_tps = standard_results['summary_stats']['tokens_per_second']['mean']
        flash_tps = flash_results['summary_stats']['tokens_per_second']['mean']
        print(f"\nGeneration Speed:")
        print(f"  Standard: {std_tps:.1f} ± {standard_results['summary_stats']['tokens_per_second']['std']:.1f} tokens/s")
        print(f"  Flash:    {flash_tps:.1f} ± {flash_results['summary_stats']['tokens_per_second']['std']:.1f} tokens/s")
        tps_speedup = flash_tps/std_tps if std_tps > 0 else 0
        print(f"  Speedup:  {tps_speedup:.2f}x")
        
        # Generation time comparison
        std_time = standard_results['summary_stats']['generation_time']['mean']
        flash_time = flash_results['summary_stats']['generation_time']['mean']
        print(f"\nGeneration Time:")
        print(f"  Standard: {std_time:.2f} ± {standard_results['summary_stats']['generation_time']['std']:.2f}s")
        print(f"  Flash:    {flash_time:.2f} ± {flash_results['summary_stats']['generation_time']['std']:.2f}s")
        time_speedup = std_time/flash_time if flash_time > 0 else 0
        print(f"  Speedup:  {time_speedup:.2f}x")
        
        # Memory usage comparison
        std_memory = standard_results['summary_stats']['memory_delta']['mean']
        flash_memory = flash_results['summary_stats']['memory_delta']['mean']
        print(f"\nMemory Usage:")
        print(f"  Standard: {std_memory:.3f} ± {standard_results['summary_stats']['memory_delta']['std']:.3f}GB")
        print(f"  Flash:    {flash_memory:.3f} ± {flash_results['summary_stats']['memory_delta']['std']:.3f}GB")
        memory_diff = ((flash_memory - std_memory) / abs(std_memory)) * 100 if std_memory != 0 else 0
        print(f"  Difference: {memory_diff:+.1f}%")
        
        # Overall assessment
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        if tps_speedup > 1.1:
            print(f"✅ Flash Attention provides significant speedup: {tps_speedup:.2f}x faster!")
        elif tps_speedup > 1.0:
            print(f"✓ Flash Attention provides modest speedup: {tps_speedup:.2f}x faster")
        else:
            print(f"⚠️  Flash Attention is slower in this test: {tps_speedup:.2f}x (may vary by model/hardware)")
        
        # Statistical significance check
        if standard_results['summary_stats']['tokens_per_second']['std'] > 0 and flash_results['summary_stats']['tokens_per_second']['std'] > 0:
            # Simple significance check based on standard deviations
            combined_std = (standard_results['summary_stats']['tokens_per_second']['std'] + flash_results['summary_stats']['tokens_per_second']['std']) / 2
            diff = abs(flash_tps - std_tps)
            if diff > 2 * combined_std:
                print("📊 Result appears statistically significant (>2σ difference)")
            else:
                print("📊 Result may not be statistically significant (<2σ difference)")
        
        # Save results to file
        all_results = {
            'standard_attention': standard_results,
            'flash_attention': flash_results,
            'comparison': {
                'load_time_speedup': load_speedup,
                'generation_speedup': tps_speedup,
                'time_speedup': time_speedup,
                'memory_difference_percent': memory_diff
            },
            'hardware_info': {
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'flash_attn_version': flash_attn.__version__ if 'flash_attn' in globals() else None
            }
        }
        
        save_benchmark_results(all_results)
        
        print(f"\n{'='*60}")
    else:
        print("❌ Benchmark failed - could not complete comparison")

if __name__ == "__main__":
    main()
