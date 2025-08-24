# 4-bit Quantization Support

The Simple CLI Chatbot now fully supports 4-bit quantization for efficient memory usage on RTX 5060 Ti 16GB GPU.

## How It Works

The chatbot automatically detects the `use_4bit_quantization: true` setting in model configurations and applies BitsAndBytesConfig quantization using the NF4 quantization type with double quantization for optimal performance.

## Current Configuration

### Models with 4-bit Quantization Enabled

- **CodeLlama 13B Instruct (4-bit)**: `codellama-13b-instruct-4bit`
  - Memory usage: ~8-10GB VRAM (vs ~26GB unquantized)
  - Quality: Minimal degradation for code generation tasks
  - Max Memory Limit: 15GB

### Models with Optional 4-bit Quantization

The following models can optionally enable 4-bit quantization by uncommenting the `use_4bit_quantization: true` line in their settings:

- **Llama 3.1 8B Instruct**: Memory reduction from ~15GB to ~8GB
- **Mistral 7B Instruct**: Memory reduction from ~12GB to ~6GB

## Enabling 4-bit Quantization

To enable 4-bit quantization for any model, edit `config/models.yaml`:

```yaml
models:
  your-model-name:
    # ... other config ...
    settings:
      # ... other settings ...
      use_4bit_quantization: true  # Enable 4-bit quantization
      max_memory_gb: 15  # Set appropriate memory limit
```

## Benefits for RTX 5060 Ti

### Memory Efficiency
- **13B models**: Can run comfortably with 8-10GB VRAM instead of 26GB
- **8B models**: Reduced from 15GB to ~8GB
- **7B models**: Reduced from 12GB to ~6GB

### Performance
- **Loading time**: Slightly longer initial load due to quantization process
- **Inference speed**: Minimal impact, still very fast on RTX 5060 Ti
- **Quality**: Minimal degradation for most tasks

### Use Cases
- **Large models on 16GB GPU**: Enables running 13B models that normally require 24GB+
- **Multi-model scenarios**: Can load multiple smaller quantized models simultaneously
- **Development**: Faster iteration when working with large models

## Technical Details

### Quantization Settings Used
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Memory Management
- **Device mapping**: Automatic with memory limits
- **Max memory**: Configurable per model (e.g., 15GB for 13B models)
- **GPU placement**: Handled automatically by accelerate library

## Testing Quantization

Use the included test script to verify quantization settings:

```bash
# Test all model configurations
python test_quantization.py

# Test specific quantized model
python test_quantization.py codellama-13b-instruct-4bit
```

## Troubleshooting

### Common Issues

1. **Loading errors with quantized models**:
   - Ensure `bitsandbytes` is properly installed
   - Check CUDA compatibility
   - Verify sufficient system RAM

2. **Memory still too high**:
   - Reduce `max_memory_gb` setting
   - Enable 4-bit quantization for additional models
   - Close other GPU applications

3. **Performance degradation**:
   - 4-bit quantization has minimal quality impact
   - For critical applications, consider using smaller unquantized models
   - Adjust temperature and other generation parameters

### Verification Commands

```bash
# Check GPU memory usage
nvidia-smi

# Monitor during model loading
watch -n 1 nvidia-smi

# Check quantization in logs
grep "quantization" chatbot.log
```

## Recommendations for RTX 5060 Ti

### Optimal Configuration
- **Small models (1-3B)**: Use FP16 for best quality
- **Medium models (7-8B)**: FP16 works well, 4-bit optional for memory savings
- **Large models (13B+)**: Use 4-bit quantization (required for 16GB GPU)

### Memory Budget
- **System RAM**: 16GB+ recommended
- **GPU VRAM**: 16GB RTX 5060 Ti
- **Reserved memory**: ~2GB for system and other processes
- **Available for models**: ~14GB effective

This setup allows you to run much larger models than would normally fit on a 16GB GPU while maintaining excellent performance and quality.
