# 4-bit and 8-bit Quantization Support

The Simple CLI Chatbot now fully supports both 4-bit and 8-bit quantization for efficient memory usage on RTX 5060 Ti 16GB GPU.

## How It Works

The chatbot automatically detects quantization settings in model configurations:
- `use_4bit_quantization: true` - Applies BitsAndBytesConfig with NF4 quantization
- `use_8bit_quantization: true` - Applies BitsAndBytesConfig with 8-bit quantization

**Priority**: 4-bit quantization takes precedence over 8-bit if both are enabled.

## Current Configuration

### Models with Quantization Enabled

**4-bit Quantization (Maximum Memory Savings):**
- **CodeLlama 13B Instruct (4-bit)**: `codellama-13b-instruct-4bit`
  - Memory usage: ~8-10GB VRAM (vs ~26GB unquantized)
  - Quality: Minimal degradation for code generation tasks
  - Max Memory Limit: 15GB

- **Llama 3.1 8B Instruct (4-bit)**: `llama-3.1-8b-instruct-4bit`  
  - Memory usage: ~6-8GB VRAM (vs ~15GB unquantized)
  - Quality: Slight degradation, excellent memory efficiency
  - Max Memory Limit: 14GB

**8-bit Quantization (Balanced Performance):**
- **CodeLlama 13B Instruct (8-bit)**: `codellama-13b-instruct-8bit`
  - Memory usage: ~12-14GB VRAM (vs ~26GB unquantized)
  - Quality: Better than 4-bit, moderate memory savings
  - CPU Offload: Enabled for stability

- **Llama 3.1 8B Instruct (8-bit)**: `llama-3.1-8b-instruct-8bit`
  - Memory usage: ~10-12GB VRAM (vs ~15GB unquantized)  
  - Quality: Better than 4-bit, good memory efficiency
  - CPU Offload: Enabled for stability

### Models with Optional Quantization

The following models can optionally enable quantization by uncommenting the appropriate line in their settings:

- **Llama 3.1 8B Instruct**: Choose between FP16, 8-bit, or 4-bit
- **Mistral 7B Instruct**: Choose between FP16, 8-bit, or 4-bit
- **CodeLlama 7B models**: Choose between FP16, 8-bit, or 4-bit

## Quantization Comparison

| Type | Memory Savings | Quality | Loading Time | Best For |
|------|----------------|---------|--------------|----------|
| **FP16** | None | Best | Fast | Small-medium models, high quality needs |
| **8-bit** | ~40-50% | Good | Moderate | Balanced performance and memory |
| **4-bit** | ~60-70% | Fair | Slow | Large models, memory-constrained |

## Enabling Quantization

To enable quantization for any model, edit `config/models.yaml`:

```yaml
models:
  your-model-name:
    # ... other config ...
    settings:
      # ... other settings ...
      # Choose ONE of the following:
      use_8bit_quantization: true   # Better quality, moderate memory savings
      use_4bit_quantization: true   # Maximum memory savings
      max_memory_gb: 15  # Set appropriate memory limit
```

**Important**: 4-bit takes precedence over 8-bit if both are enabled.

## Benefits for RTX 5060 Ti

### Memory Efficiency
- **13B models with 4-bit**: Can run comfortably with 8-10GB VRAM instead of 26GB
- **13B models with 8-bit**: Use 12-14GB VRAM instead of 26GB  
- **8B models with 4-bit**: Reduced from 15GB to ~6-8GB
- **8B models with 8-bit**: Reduced from 15GB to ~10-12GB

### Performance
- **Loading time**: 4-bit slower, 8-bit moderate overhead
- **Inference speed**: Minimal impact on RTX 5060 Ti for both types
- **Quality**: 8-bit > 4-bit > minimal degradation for most tasks

### Use Cases
- **4-bit**: Large models on 16GB GPU, maximum memory efficiency
- **8-bit**: Balanced performance and memory, better quality than 4-bit
- **Multi-model scenarios**: Can load multiple quantized models simultaneously
- **Development**: Faster iteration when working with large models

## Technical Details

### 4-bit Quantization Settings
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 8-bit Quantization Settings
```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_threshold=6.0
)
```

### Memory Management
- **4-bit**: Strict GPU memory limits, device mapping
- **8-bit**: Flexible memory with CPU offload support
- **Max memory**: Configurable per model (12-15GB for large models)
- **GPU placement**: Handled automatically by accelerate library

## Testing Quantization

Use the included test script to verify quantization settings:

```bash
# Test all model configurations
python test_quantization.py

# Test specific quantized models
python test_quantization.py codellama-13b-instruct-4bit
python test_quantization.py codellama-13b-instruct-8bit
python test_quantization.py llama-3.1-8b-instruct-4bit
python test_quantization.py llama-3.1-8b-instruct-8bit
```

## Troubleshooting

### Common Issues

1. **8-bit loading errors with CPU offload**:
   - This is normal: "Some parameters are on the meta device because they were offloaded to the cpu"
   - Enables larger models to fit in 16GB VRAM
   - Slight performance impact for offloaded layers

2. **4-bit loading slower than expected**:
   - Quantization process takes time during loading
   - Consider 8-bit for faster loading with moderate memory savings

3. **Memory still too high**:
   - Try 4-bit instead of 8-bit for maximum savings
   - Reduce `max_memory_gb` setting  
   - Close other GPU applications

4. **Quality degradation**:
   - 8-bit generally better quality than 4-bit
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
- **Medium models (7B)**: FP16 or 8-bit depending on memory needs
- **Large models (8B)**: 8-bit recommended, 4-bit for maximum efficiency  
- **Extra large models (13B+)**: 8-bit or 4-bit quantization (required for 16GB GPU)

### Memory Budget Strategy
- **Conservative (12GB effective)**: Use 8-bit for large models
- **Aggressive (14GB effective)**: Use 4-bit for maximum model size
- **Balanced**: Mix of FP16 smaller models and quantized larger models

This setup allows you to run much larger models than would normally fit on a 16GB GPU while maintaining excellent performance and quality with the choice between 4-bit (maximum efficiency) and 8-bit (balanced performance).
