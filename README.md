# 🎙️ ParaSync

> **Paralinguistic Synchronization Toolkit**
>
> 高精度语音对齐工具包，支持非语言事件（呼吸、笑声等）与音素序列的协同对齐

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
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
| 🈯 **中英文支持** | 中文拼音声韵母拆分，英文字母级对齐 |
| 🚀 **一键处理** | 端到端 pipeline，从音频到 TextGrid |

---

## 📁 项目结构

```
parasync/
├── aligner/
│   ├── __init__.py              # 模块初始化
│   ├── phoneme_aligner.py       # 🎯 MMS 强制对齐核心
│   ├── nonverbal_detector.py    # 🫁 非语言事件检测
│   ├── asr_recognizer.py        # 🎤 ASR 语音识别
│   ├── textgrid_exporter.py     # 📊 TextGrid 导出器
│   ├── mms_aligner.py           # MMS-1B 对齐器（实验性）
│   └── torchaudio_aligner.py    # Torchaudio 对齐器
├── parasync.py                  # 🚀 主入口 CLI
├── process_audio.py             # 🎵 完整处理流程
├── example.py                   # 💡 使用示例
├── tests/                       # 🧪 单元测试
│   ├── conftest.py
│   ├── test_phoneme_aligner.py
│   └── ...
├── examples/                    # 📂 示例数据
│   ├── Englishtest.wav          # 英文测试音频
│   ├── Englishtest.TextGrid     # 英文对齐结果
│   └── ...
├── requirements.txt             # 📦 依赖列表
└── README.md                    # 📖 项目文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 完整处理流程（推荐）

```bash
# 中文音频处理
uv run python process_audio.py audio.wav \
  -t "你好，我是赵君君" \
  -l zh \
  -o output/

# 英文音频处理（自动 ASR 识别）
uv run python process_audio.py Englishtest.wav \
  -o output/ \
  -l en
```

### 3. 使用 ASR + 对齐

```bash
# 提供文本进行对齐
uv run python process_audio.py audio.wav \
  -t "你好世界" \
  -l zh \
  -o output/

# 自动 ASR 识别后对齐（不提供 -t 参数）
uv run python process_audio.py audio.wav \
  -l zh \
  -o output/
```

### 4. 单独功能

```bash
# 仅音素对齐
python parasync.py align \
  -a audio.wav \
  -t "你好世界" \
  -l zh \
  -o result.json

# 仅非语言事件检测
python parasync.py detect \
  -a audio_with_breath.wav \
  -o events.json

# 完整 pipeline
python parasync.py pipeline \
  -a audio.wav \
  -t "你好，[hx] 我是赵君君" \
  -l zh \
  -o result.TextGrid
```

---

## 📖 使用示例

### 测试用例 1：英文对齐

```bash
# 处理英文音频
uv run python process_audio.py Englishtest.wav -o . -l en

# 输出：
# 🎵 音频: Englishtest.wav
# 🔍 Step 1/4: ASR 语音识别...
#    识别结果: "My name is Zhao Junjun. Ha ha ha!"
#    置信度: 0.90
# 🎯 Step 2/4: 音素对齐...
#    对齐完成: 18 个音素段
# 👂 Step 3/4: 非语言事件检测...
#    检测到 4 个事件
# ✅ 导出成功: Englishtest.TextGrid
```

### 测试用例 2：中文对齐

```bash
# 处理中文音频
uv run python process_audio.py test01.wav \
  -t "我是赵君君哈哈" \
  -l zh \
  -o .
```

### API 使用示例

```python
from aligner import PhonemeAligner

# 初始化对齐器
aligner = PhonemeAligner(lang="zh")

# 执行对齐
segments = aligner.align_text("audio.wav", "你好世界")

for seg in segments:
    print(f"{seg.token}: {seg.start_time:.3f}s - {seg.end_time:.3f}s")
```

---

## 🎯 可视化界面

启动 Web 可视化界面查看对齐结果：

```bash
# 启动 HTTP 服务器
uv run python -m http.server 8000

# 打开浏览器访问
open http://localhost:8000/viewer.html
```

或者直接打开 `viewer.html` 文件查看示例。

---

## 📊 TextGrid 四层结构

ParaSync 生成的 TextGrid 包含四个层级：

| 层级 | 名称 | 内容示例 | 用途 |
|------|------|---------|------|
| 1 | **Event** | `[hx]`, `[laugh]`, `[cry]` | 非语言事件标注 |
| 2 | **Word** | `你好`, `M` | 汉字/英文字母 |
| 3 | **Phoneme-CN** | `zh`, `un` | 中文声韵母拆分 |
| 4 | **IPA-EN** | - | 国际音标（预留） |

### 英文对齐示例

```
IntervalTier: Word
  2.200000 2.340000  Y
  2.340000 2.420000  N
  2.420000 2.480000  A
  2.480000 2.520000  M
  2.520000 2.700000  E
  ...
```

### 中文对齐示例

```
IntervalTier: Word
  0.000000 1.940000  我
  1.980000 2.060000  是
  2.100000 2.280000  赵
  2.280000 2.680000  君
  2.740000 4.900000  哈

IntervalTier: Phoneme-CN
  0.000000 0.582000  w
  0.582000 1.940000  o
  ...
```

---

## 🧪 测试

```bash
# 运行全部测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_phoneme_aligner.py -v

# 生成覆盖率报告
uv run pytest --cov=parasync --cov-report=html
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
│  │  (Wav2Vec2 CTC   │    │   • Energy-based detection │     │
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

## 📝 更新日志

### 最新更新

- ✅ **英文对齐修复**: 修复了英文文本处理，现在正确显示大写字母
- ✅ **ASR 集成**: 添加 Whisper ASR，支持自动语音识别后对齐
- ✅ **MMS-1B 实验**: 尝试集成 MMS-1B（14GB），因模型类型不匹配暂时搁置
- ✅ **Python 3.10**: 降级到 Python 3.10 以兼容 fairseq
- ✅ **单元测试**: 添加完整的 pytest 测试套件

### 已知问题

- MMS-1B 模型（14GB）是英语专用微调版本，不支持中文
- 需要下载正确的多语言 MMS-1B 模型才能使用 MMS FA

详见 [MMS_STATUS.md](MMS_STATUS.md)

---

## 🔧 关键技术细节

### 中文拼音声韵母拆分

```python
# 输入: "zhun"
# 输出: 声母=["zh"], 韵母=["un"]

# 输入: "zhuang"
# 输出: 声母=["zh"], 韵母=["uang"]
```

### 非语言事件检测特征

| 事件类型 | 主要特征 |
|---------|---------|
| 呼吸 `[hx]` | 低频能量 + 中等强度 + 低过零率 |
| 笑声 `[laugh]` | 能量波动 + 高频过零率 + 频谱变化 |
| 静音 `[sil]` | 极低能量 |

---

## 🛣️ 路线图

- [x] MMS 强制对齐核心
- [x] 非语言事件检测（启发式）
- [x] TextGrid 多层级导出
- [x] ASR 语音识别集成
- [x] 单元测试套件
- [x] Web 可视化界面
- [ ] YAMNet/PANNs 深度学习检测
- [ ] SSML 输入解析支持
- [ ] 批量处理模式

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [Facebook MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - 多语言语音模型
- [OpenAI Whisper](https://github.com/openai/whisper) - ASR 模型
- [torchaudio](https://pytorch.org/audio/stable/index.html) - 音频处理库
- [TextGrid](https://github.com/kylebgorman/textgrid) - Praat 格式支持
