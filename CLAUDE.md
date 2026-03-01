# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParaSync is a speech alignment toolkit that performs forced alignment with nonverbal event detection. It aligns phonemes with audio using Facebook's MMS models, detects events like breaths `[hx]` and laughter `[laugh]`, and exports multi-tier TextGrid files for Praat.

## Development Commands

### Installation

```bash
# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all unit tests (excluding slow TTS tests)
uv run pytest tests/ -m "not slow"

# Run specific test modules
uv run pytest tests/test_merge_alignments.py -v
uv run pytest tests/test_textgrid_exporter.py -v

# Run with coverage report
uv run pytest --cov=parasync --cov=aligner --cov-report=html

# Run integration tests (requires TTS model)
uv run pytest tests/test_integration.py -v
```

### Running the Pipeline

```bash
# Run full alignment pipeline
python parasync.py pipeline -a audio.wav -t "你好世界" -l zh -o result.TextGrid

# Run individual components
python parasync.py align -a audio.wav -t "你好世界" -l zh -o output.json
python parasync.py detect -a audio.wav -m heuristic -o events.json

# Run examples (without audio file, shows structure only)
python example.py

# Generate test audio with ChatTTS
python -c "from parasync.tts_generator import ChatTTSGenerator; \
  g = ChatTTSGenerator(); \
  g.generate('你好[uv_break]世界', 'test.wav')"
```

## Architecture

### Core Components

**1. ChatTTSGenerator** (`parasync/tts_generator.py`)
- Generates test audio with ChatTTS model
- Supports special tags: `[uv_break]`, `[laugh]`, `[lbreak]`, `[break]`
- Methods:
  - `generate()`: Single utterance generation
  - `generate_test_suite()`: Batch generation for test scenarios
  - `generate_with_events()`: Insert events at specific positions
- Sample rate: 24kHz (ChatTTS default)
- Used for: Integration testing, validation of event detection accuracy

**2. PhonemeAligner** (`aligner/phoneme_aligner.py`)
- Uses Facebook MMS TTS models (`facebook/mms-tts-zho`, etc.) for CTC forced alignment
- Implements a custom CTC forced aligner using dynamic programming (Viterbi algorithm)
- Flow: audio preprocessing → emission computation → CTC alignment → segment extraction
- Supports 8+ languages via `LANG_MODEL_MAP`
- Key method: `ctc_forced_align()` implements the DP alignment with blank token handling

**3. NonverbalEventDetector** (`aligner/nonverbal_detector.py`)
- Two methods: `heuristic` (energy/spectral features) or `model` (YAMNet - not fully implemented)
- Heuristic detection extracts: RMS energy, spectral centroid, zero-crossing rate, spectral rolloff, mel spectrogram
- Event detection masks:
  - Breath: `0.05 < energy < 0.3` AND `centroid < 0.15 * sr` AND `zcr < mean_zcr`
  - Laugh: `energy > 0.3` AND `energy_variation > 70th percentile` AND `zcr > 40th percentile`
  - Silence: `energy < 0.005`

**4. merge_alignments** (`aligner/nonverbal_detector.py`)
- Critical fusion logic that resolves temporal conflicts between phonemes and events
- Priority rule: **events override phonemes** on overlap
- Implementation: sorts all segments by start time, then truncates overlapping phonemes to make room for events

**5. TextGridExporter** (`aligner/textgrid_exporter.py`)
- Creates 4-tier TextGrid: Event → Word → Phoneme-CN → IPA-EN
- Chinese-specific: splits pinyin into initials/finals (e.g., "zhun" → "zh" + "un")
- Time allocation: initial gets 30% of unit duration, final gets 70%
- IPA conversion via lookup tables for both Chinese and English

### Data Flow

**Production Pipeline:**
```
audio.wav + transcript
    ↓
PhonemeAligner (MMS CTC) ──┐
                           ├──→ merge_alignments ──→ TextGridExporter ──→ .TextGrid
detector.detect() ──────────┘      (conflict resolution)
```

**Test/Development Pipeline:**
```
ChatTTSGenerator ──→ audio.wav ──┐
                                  ├──→ PhonemeAligner + detector ──→ Validation
transcript with tags ────────────┘
```

### Key Data Structures

```python
# AlignmentSegment (from phoneme_aligner)
@dataclass
class AlignmentSegment:
    token: str        # phoneme or character
    start_time: float
    end_time: float
    confidence: float

# NonverbalEvent (from nonverbal_detector)
@dataclass
class NonverbalEvent:
    event_type: EventType  # BREATH("hx"), LAUGH("laugh"), etc.
    start_time: float
    end_time: float
    confidence: float

# Merged segment (from merge_alignments)
{
    "type": "phoneme" | "event",
    "token": str,
    "start": float,
    "end": float,
    "confidence": float,
    "event_type": str  # only for events
}
```

## Implementation Notes

### CTC Forced Alignment Algorithm
The `ctc_forced_align()` method implements Viterbi alignment for CTC:
1. Constructs CTC token sequence with blanks between labels
2. DP matrix tracks best path probability to each (token, time) position
3. Backtrace pointers record optimal transitions
4. Supports blank-skipping transitions (jumping from i to i-2)

### Audio Feature Extraction
Uses librosa with 10ms hop length:
- `rms`: Frame energy for silence/activity detection
- `spectral_centroid`: Frequency "center of mass" (low for breath, varied for laugh)
- `zcr`: Zero-crossing rate (noise vs. tonal discrimination)
- Frame duration calculated as `hop_length / sample_rate`

## Test Suite

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_tts_generator.py       # ChatTTS generation tests (marked: tts, slow)
├── test_phoneme_aligner.py     # Phoneme alignment tests
├── test_nonverbal_detector.py  # Event detection tests
├── test_merge_alignments.py    # Critical fusion logic tests
├── test_textgrid_exporter.py   # TextGrid export tests
└── test_integration.py         # End-to-end pipeline tests
```

### Key Fixtures (conftest.py)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `chattts_generator` | session | Shared ChatTTS model instance |
| `generated_audio_suite` | module | Pre-generated test audio files |
| `sample_aligner` | function | Mock PhonemeAligner with patched model loading |
| `sample_detector` | function | Fresh NonverbalEventDetector instance |
| `sample_audio_file` | function | Synthetic sine wave audio for testing |
| `mock_phoneme_segments` | function | Mock phoneme alignment data |
| `mock_event_segments` | function | Mock nonverbal event data |

### Test Markers

- `slow`: Tests requiring model loading (TTS, MMS)
- `integration`: End-to-end tests
- `tts`: Tests requiring ChatTTS model

### Critical Test Scenarios

**1. Event Override Priority** (`test_merge_alignments.py`)
```python
# Events must override phonemes on temporal overlap
merged = merge_alignments(phonemes, events)
# Verify: no overlapping time intervals in result
```

**2. Pinyin Splitting** (`test_textgrid_exporter.py`)
```python
# Verify correct initial/final decomposition
"zhun" → ["zh"] + ["un"]
"hao" → ["h"] + ["ao"]
```

**3. Event Detection Accuracy** (`test_integration.py`)
```python
# Generate audio with known tags, verify detection
audio = generate("你好[uv_break]世界")
events = detector.detect(audio)
# Expect: BREATH event around position of [uv_break]
```

### Test Results Summary

| Module | Tests | Status |
|--------|-------|--------|
| test_merge_alignments.py | 20 | ✅ All pass |
| test_textgrid_exporter.py | 34 | ✅ All pass |
| test_tts_generator.py | 7 | ✅ All pass |
| test_integration.py | 3 | ✅ All pass |
| test_nonverbal_detector.py | 25 | ⚠️ 3 env failures |
| test_phoneme_aligner.py | 16 | ⚠️ 2 env failures |

**Environment-related failures**: FFmpeg/torchcodec library loading issues on macOS (not code issues)

### Chinese Text Processing
- `jieba` for segmentation (optional, falls back to character-level)
- MMS processor handles character-to-token mapping
- Pinyin splitting uses `INITIALS` set ordered by length (to match "zh" before "z")

### Merge Conflict Resolution
In `merge_alignments()`, when `seg["start"] < last["end"] - threshold`:
- If incoming is event and last is phoneme: truncate last phoneme's end time
- If incoming is phoneme and last is event: delay phoneme's start time
- Same type: keep higher confidence segment

### TextGrid Tier Structure
All tiers are `IntervalTier` (not PointTier), with intervals containing:
- Event tier: `[hx]`, `[laugh]`, `[cry]`, `[sil]`
- Word tier: Chinese characters or English words
- Phoneme-CN: Split pinyin (initials/finals as separate intervals)
- IPA-EN: IPA symbols converted via lookup tables
