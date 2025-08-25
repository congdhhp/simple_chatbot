# Flash Attention Integration

## Tóm tắt

Đã thành công integrate **Flash Attention 2** vào Simple Chatbot project để tối ưu hóa hiệu suất trên GPU RTX 5060 Ti 16GB.

## Những thay đổi đã thực hiện

### 1. ModelManager (src/model_manager.py)

**Thêm Flash Attention support:**
- Import flash_attn library với error handling
- Thêm method `_configure_flash_attention()` để kiểm tra và cấu hình
- Tự động detect GPU compute capability và compatibility
- Set `attn_implementation="flash_attention_2"` khi load model
- Thêm Flash Attention status vào model info

**Key features:**
```python
# Tự động phát hiện Flash Attention
FLASH_ATTN_AVAILABLE = True/False

# Cấu hình dựa trên model settings
use_flash_attention: true  # từ config YAML

# Set implementation khi load model
model_kwargs['attn_implementation'] = "flash_attention_2"
```

### 2. CLI Interface (src/cli.py)

**Enhanced model info display:**
- Hiển thị Flash Attention status với emoji indicators
- Show availability và config status
- Visual feedback: ✅ Enabled, ❌ Disabled, ❌ Not Available

### 3. Configuration (requirements.txt)

**Uncommented Flash Attention:**
```
flash-attn>=2.3.0  # Previously commented out
```

### 4. Test Scripts

**Tạo 3 test scripts:**
1. `test_flash_attention.py` - Comprehensive testing với full model loading
2. `benchmark_flash_attention.py` - Performance comparison tool
3. `simple_flash_test.py` - Quick verification script

## Cách sử dụng

### Kiểm tra Flash Attention status:
```bash
# Activate venv
source venv/bin/activate

# Run simple test
python simple_flash_test.py

# Run comprehensive test
python test_flash_attention.py

# Benchmark performance (so sánh với standard attention)
python benchmark_flash_attention.py
```

### Trong chatbot:
```bash
python chatbot.py -m llama-3.2-1b-instruct
# Gõ /info để xem Flash Attention status
```

## Configuration trong models.yaml

Mọi model đều có option:
```yaml
settings:
  use_flash_attention: true  # Enable Flash Attention
  max_memory_gb: 14         # Memory limit
```

## Lợi ích của Flash Attention

### 1. **Memory Efficiency**
- Giảm memory footprint trong attention computation
- Cho phép sử dụng longer sequences với cùng VRAM

### 2. **Speed Improvement**
- Faster attention computation trên GPU hiện đại
- Optimized cho architectures với compute capability >= 8.0

### 3. **Automatic Detection**
- Tự động detect hardware compatibility
- Fallback về standard attention nếu không support

## Hardware Requirements

### Optimal:
- GPU với compute capability >= 8.0 (RTX 30xx, RTX 40xx, RTX 50xx)
- CUDA support
- Sufficient VRAM

### Fallback behavior:
- GPU compute < 8.0: Warning nhưng vẫn sử dụng Flash Attention
- CUDA không available: Tự động disable
- Flash-attn không install: Graceful fallback về eager attention

## Kết quả Test

```
Flash Attention Integration Test
==================================================
Model Information:
  Name: Llama 3.2 1B Instruct  
  Device: cuda:0
  Flash Attention: ✅ Enabled
  Flash Attention Available: True
  Flash Attention Config: True

Test completed successfully!
```

## Status trong Model Info

Khi gõ `/info` trong chatbot:
```
┌─ Current Model Info ─┐
│ Model: Llama 3.2 1B  │
│ Device: cuda:0       │
│ Flash Attention: ✅   │
│ Flash Attention Config: ✅ │
└──────────────────────┘
```

## Implementation Details

### Smart Configuration:
- Chỉ enable khi `use_flash_attention: true` trong config
- Automatic hardware compatibility check
- Graceful degradation nếu không support

### Error Handling:
- Try/catch cho flash_attn import
- Warning messages cho incompatible hardware
- Fallback mechanisms

### Integration:
- Seamless integration với existing model loading
- No breaking changes cho existing functionality
- Backward compatible

Flash Attention integration đã hoàn thành và ready for production use!
