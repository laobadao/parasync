# 🎙️ ParaSync

> **Paralinguistic Synchronization Toolkit**
>
> 高精度语音对齐工具包，支持非语言事件（呼吸、笑声等）与音素序列的协同对齐

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🎯 **MMS 强制对齐** | 基于 Facebook MMS 模型的高精度音素对齐 |
| 🫁 **非语言检测** | 呼吸声 `[hx]`、笑声 `[laugh]`、哭声 `[cry]` 等事件识别 |
| 🔀 **智能融合** | 音素与非语言事件的时序融合与冲突解决 |
| 📊 **多层级导出** | Event / Word / Phoneme-CN / IPA-EN 四层 TextGrid |
| 🈯 **中英文支持** | 中文拼音声韵母拆分，英文 IPA 音标转换 |
| 🚀 **一键处理** | 端到端 pipeline，从音频到 TextGrid |

---

## 📁 项目结构

```
parasync/
├── aligner/
│   ├── __init__.py              # 模块初始化
│   ├── phoneme_aligner.py       # 🎯 MMS 强制对齐核心
│   ├── nonverbal_detector.py    # 🫁 非语言事件检测
│   └── textgrid_exporter.py     # 📊 TextGrid 导出器
├── parasync.py                  # 🚀 主入口 CLI
├── example.py                   # 💡 使用示例
├── requirements.txt             # 📦 依赖列表
└── README.md                    # 📖 项目文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础对齐

```bash
python parasync.py align \
  -a audio.wav \
  -t "你好世界，我是 ParaSync" \
  -l zh \
  -o result.json
```

### 3. 非语言事件检测

```bash
python parasync.py detect \
  -a audio_with_breath.wav \
  -o events.json
```

### 4. 完整流程（推荐）

```bash
python parasync.py pipeline \
  -a audio.wav \
  -t "你好，[hx] 我是赵君君" \
  -l zh \
  -o result.TextGrid
```

---

## 📖 API 使用示例

### 基础音素对齐

```python
from aligner import PhonemeAligner

# 初始化对齐器
aligner = PhonemeAligner(lang="zh")

# 执行对齐
segments = aligner.align_text("audio.wav", "你好世界")

for seg in segments:
    print(f"{seg.token}: {seg.start_time:.3f}s - {seg.end_time:.3f}s")
```

### 非语言事件检测

```python
from aligner import NonverbalEventDetector

detector = NonverbalEventDetector(method="heuristic")
events = detector.detect("audio.wav")

for evt in events:
    print(f"[{evt.event_type.value}]: {evt.start_time:.3f}s")
```

### 融合与导出

```python
from aligner.nonverbal_detector import merge_alignments
from aligner import TextGridExporter

# 融合对齐结果
merged = merge_alignments(phoneme_segments, events)

# 导出 TextGrid
exporter = TextGridExporter()
exporter.export_from_alignment(merged, "result.TextGrid")
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Input                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Audio File  │  │  Transcript  │  │  [hx] [laugh]   │   │
│  │   (.wav)     │  │  (Text/SSML) │  │   Labels        │   │
│  └──────┬───────┘  └──────┬───────┘  └─────────────────┘   │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    ParaSync Core                             │
│  ┌──────────────────┐    ┌────────────────────────────┐     │
│  │  PhonemeAligner  │    │   NonverbalEventDetector   │     │
│  │  (MMS-Forced     │    │   • Energy-based detection │     │
│  │   Alignment)     │    │   • Spectral features      │     │
│  │                  │    │   • [hx] [laugh] [cry]     │     │
│  └────────┬─────────┘    └────────────┬───────────────┘     │
│           │                           │                      │
│           └───────────┬───────────────┘                      │
│                       ▼                                      │
│           ┌──────────────────────┐                          │
│           │   merge_alignments   │  ◄── Conflict resolution │
│           │   (Smart fusion)     │                          │
│           └──────────┬───────────┘                          │
└──────────────────────┼──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Event     │ │    Word     │ │  Phoneme    │            │
│  │   Tier      │ │    Tier     │ │    Tier     │            │
│  │ [hx][laugh] │ │  你好 世界  │ │ zh un h ao  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────────────────────────────────────┐            │
│  │   result.TextGrid (4-tier format)           │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 TextGrid 四层结构

ParaSync 生成的 TextGrid 包含四个层级：

| 层级 | 名称 | 内容示例 | 用途 |
|------|------|---------|------|
| 1 | **Event** | `[hx]`, `[laugh]`, `[cry]` | 非语言事件标注 |
| 2 | **Word** | `你好`, `世界` | 汉字/英文单词 |
| 3 | **Phoneme-CN** | `zh`, `un`, `h`, `ao` | 中文声韵母拆分 |
| 4 | **IPA-EN** | `tʂ`, `u`, `n` | 国际音标（英文） |

### 示例输出

```
IntervalTier: Event
  0.500000 0.800000  [hx]
  2.000000 2.300000  [laugh]

IntervalTier: Word
  0.000000 0.800000  你好
  0.800000 1.500000  世界

IntervalTier: Phoneme-CN
  0.000000 0.120000  zh
  0.120000 0.400000  un
  0.400000 0.550000  h
  0.550000 0.800000  ao
```

---

## 🔧 关键技术细节

### 中文拼音声韵母拆分

```python
# 输入: "zhun"
# 输出: 声母=["zh"], 韵母=["un"]

# 输入: "zhuang"
# 输出: 声母=["zh"], 韵母=["uang"]
```

支持 23 个声母和完整的韵母表，可精确拆分复杂拼音如 `chuang`、`zhun` 等。

### 非语言事件检测特征

| 事件类型 | 主要特征 |
|---------|---------|
| 呼吸 `[hx]` | 低频能量 + 中等强度 + 低过零率 |
| 笑声 `[laugh]` | 能量波动 + 高频过零率 + 频谱变化 |
| 静音 `[sil]` | 极低能量 |

### 冲突解决策略

当非语言事件与音素时间重叠时：

```python
# 优先级: 非语言事件 > 音素
if event overlaps with phoneme:
    # 截断音素，保留事件
    phoneme.end = event.start
```

---

## 🛣️ 路线图

- [x] MMS 强制对齐核心
- [x] 非语言事件检测（启发式）
- [x] TextGrid 多层级导出
- [ ] YAMNet/PANNs 深度学习检测
- [ ] SSML 输入解析支持
- [ ] 批量处理模式
- [ ] Web 可视化界面

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [Facebook MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - 多语言语音模型
- [torchaudio](https://pytorch.org/audio/stable/index.html) - 音频处理库
- [TextGrid](https://github.com/kylebgorman/textgrid) - Praat 格式支持
