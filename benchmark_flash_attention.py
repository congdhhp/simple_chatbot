#!/usr/bin/env python3
"""Performance benchmark script for Flash Attention vs Standard Attention."""

import sys
import os
import time
import torch

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
    model_name = "llama-3.2-1b-instruct"
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
        
        # Test prompts of varying lengths
        test_prompts = [
            "Hello!",
            "Write a short story about a robot learning to cook.",
            "Explain the concept of machine learning and its applications in modern technology. Include examples of how it's used in different industries and discuss both benefits and potential challenges.",
        ]
        
        results = {
            'flash_attention': use_flash_attention,
            'load_time': load_time,
            'generation_times': [],
            'tokens_per_second': [],
            'memory_usage': []
        }
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTesting prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            prompt_times = []
            prompt_tps = []
            
            for run in range(num_tests):
                # Clear cache before each run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Measure memory before generation
                memory_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                # Generate response
                start_time = time.time()
                response = model_manager.generate_response(prompt)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Measure memory after generation
                memory_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                memory_used = memory_after - memory_before
                
                # Estimate tokens (rough approximation)
                response_tokens = len(response.split()) * 1.3  # Rough token estimate
                tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0
                
                prompt_times.append(generation_time)
                prompt_tps.append(tokens_per_second)
                
                print(f"  Run {run+1}: {generation_time:.2f}s, {tokens_per_second:.1f} tokens/s, {memory_used:.2f}GB memory")
            
            # Average results for this prompt
            avg_time = sum(prompt_times) / len(prompt_times)
            avg_tps = sum(prompt_tps) / len(prompt_tps)
            avg_memory = memory_used  # Use last measurement
            
            results['generation_times'].append(avg_time)
            results['tokens_per_second'].append(avg_tps)
            results['memory_usage'].append(avg_memory)
            
            print(f"  Average: {avg_time:.2f}s, {avg_tps:.1f} tokens/s")
        
        # Calculate overall averages
        results['avg_generation_time'] = sum(results['generation_times']) / len(results['generation_times'])
        results['avg_tokens_per_second'] = sum(results['tokens_per_second']) / len(results['tokens_per_second'])
        results['avg_memory_usage'] = sum(results['memory_usage']) / len(results['memory_usage'])
        
        print(f"\nOverall Average: {results['avg_generation_time']:.2f}s, {results['avg_tokens_per_second']:.1f} tokens/s")
        
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
    
    # Check if Flash Attention is available
    try:
        import flash_attn
        print(f"Flash Attention version: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention not available. Please install flash-attn.")
        return
    
    # Benchmark both configurations
    num_tests = 2  # Reduce for faster testing
    
    # Test standard attention
    standard_results = benchmark_attention(use_flash_attention=False, num_tests=num_tests)
    
    # Test Flash Attention
    flash_results = benchmark_attention(use_flash_attention=True, num_tests=num_tests)
    
    # Compare results
    if standard_results and flash_results:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        print(f"Load Time:")
        print(f"  Standard: {standard_results['load_time']:.2f}s")
        print(f"  Flash:    {flash_results['load_time']:.2f}s")
        print(f"  Speedup:  {standard_results['load_time']/flash_results['load_time']:.2f}x")
        
        print(f"\nGeneration Speed:")
        print(f"  Standard: {standard_results['avg_tokens_per_second']:.1f} tokens/s")
        print(f"  Flash:    {flash_results['avg_tokens_per_second']:.1f} tokens/s")
        print(f"  Speedup:  {flash_results['avg_tokens_per_second']/standard_results['avg_tokens_per_second']:.2f}x")
        
        print(f"\nGeneration Time:")
        print(f"  Standard: {standard_results['avg_generation_time']:.2f}s")
        print(f"  Flash:    {flash_results['avg_generation_time']:.2f}s")
        print(f"  Speedup:  {standard_results['avg_generation_time']/flash_results['avg_generation_time']:.2f}x")
        
        print(f"\nMemory Usage:")
        print(f"  Standard: {standard_results['avg_memory_usage']:.2f}GB")
        print(f"  Flash:    {flash_results['avg_memory_usage']:.2f}GB")
        
        if flash_results['avg_tokens_per_second'] > standard_results['avg_tokens_per_second']:
            print(f"\n✅ Flash Attention provides {flash_results['avg_tokens_per_second']/standard_results['avg_tokens_per_second']:.2f}x speedup!")
        else:
            print(f"\n⚠️  Flash Attention is slower in this test (may vary by model/hardware)")
        
        print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
