# ParaSync 测试套件

## 测试结构

```
tests/
├── conftest.py                 # pytest 共享 fixtures 和配置
├── test_tts_generator.py       # TTS 语音生成测试
├── test_phoneme_aligner.py     # 音素对齐单元测试
├── test_nonverbal_detector.py  # 非语言事件检测测试
├── test_merge_alignments.py    # 融合逻辑测试（关键测试）
├── test_textgrid_exporter.py   # TextGrid 导出测试
└── test_integration.py         # 端到端集成测试
```

## 运行测试

### 安装测试依赖

```bash
uv pip install -e ".[dev]"
```

### 运行全部测试

```bash
uv run pytest
```

### 运行特定模块

```bash
# 只运行 TTS 生成测试
uv run pytest tests/test_tts_generator.py -v

# 只运行对齐器测试
uv run pytest tests/test_phoneme_aligner.py -v

# 只运行关键融合逻辑测试
uv run pytest tests/test_merge_alignments.py -v
```

### 排除慢速测试

```bash
# 排除需要加载 TTS 模型的慢速测试
uv run pytest -m "not slow"

# 只运行单元测试（排除集成测试）
uv run pytest -m "not integration"
```

### 生成覆盖率报告

```bash
# HTML 报告
uv run pytest --cov=parasync --cov=aligner --cov-report=html

# 终端报告
uv run pytest --cov=parasync --cov=aligner --cov-report=term
```

## 测试标记说明

| 标记 | 说明 | 排除方式 |
|------|------|----------|
| `slow` | 执行缓慢的测试（如 TTS 模型加载） | `-m "not slow"` |
| `integration` | 集成测试 | `-m "not integration"` |
| `tts` | 需要 TTS 模型的测试 | `-m "not tts"` |

## Fixtures 说明

### 会话级 Fixtures（每个测试会话只创建一次）

- `chattts_generator`: 共享的 ChatTTS 生成器实例
- `project_root`: 项目根目录

### 模块级 Fixtures

- `generated_audio_suite`: 预生成的测试音频套件

### 函数级 Fixtures（每个测试函数重新创建）

- `temp_output_dir`: 临时输出目录
- `sample_audio_file`: 简单的测试音频文件
- `sample_aligner`: 预初始化的音素对齐器
- `sample_detector`: 预初始化的事件检测器

## 关键测试场景

### 1. [uv_break] 检测精度验证

```python
def test_uv_break_detection():
    """验证能检测到 ChatTTS 生成的 uv_break"""
    audio_path = generate_audio("我是[uv_break]赵君君")
    events = detector.detect(audio_path)
    breath_events = [e for e in events if e.event_type == EventType.BREATH]
    assert len(breath_events) >= 1
```

### 2. 融合后时间无重叠

```python
def test_no_overlap_after_merge():
    """验证融合后时间轴无重叠"""
    merged = merge_alignments(phonemes, events)
    for i in range(len(merged) - 1):
        assert merged[i]["end"] <= merged[i+1]["start"]
```

### 3. TextGrid Praat 兼容性

```python
def test_textgrid_readable_by_praat():
    """验证 TextGrid 可被 Praat 读取"""
    tg = textgrid.TextGrid.fromFile(output_path)
    assert tg.maxTime > 0
```
