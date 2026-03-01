# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParaSync is a speech alignment toolkit that performs forced alignment with nonverbal event detection. It aligns phonemes with audio using Facebook's MMS models, detects events like breaths `[hx]` and laughter `[laugh]`, and exports multi-tier TextGrid files for Praat.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full alignment pipeline
python parasync.py pipeline -a audio.wav -t "你好世界" -l zh -o result.TextGrid

# Run individual components
python parasync.py align -a audio.wav -t "你好世界" -l zh -o output.json
python parasync.py detect -a audio.wav -m heuristic -o events.json

# Run examples (without audio file, shows structure only)
python example.py
```

## Architecture

### Core Components

**1. PhonemeAligner** (`aligner/phoneme_aligner.py`)
- Uses Facebook MMS TTS models (`facebook/mms-tts-zho`, etc.) for CTC forced alignment
- Implements a custom CTC forced aligner using dynamic programming (Viterbi algorithm)
- Flow: audio preprocessing → emission computation → CTC alignment → segment extraction
- Supports 8+ languages via `LANG_MODEL_MAP`
- Key method: `ctc_forced_align()` implements the DP alignment with blank token handling

**2. NonverbalEventDetector** (`aligner/nonverbal_detector.py`)
- Two methods: `heuristic` (energy/spectral features) or `model` (YAMNet - not fully implemented)
- Heuristic detection extracts: RMS energy, spectral centroid, zero-crossing rate, spectral rolloff, mel spectrogram
- Event detection masks:
  - Breath: `0.05 < energy < 0.3` AND `centroid < 0.15 * sr` AND `zcr < mean_zcr`
  - Laugh: `energy > 0.3` AND `energy_variation > 70th percentile` AND `zcr > 40th percentile`
  - Silence: `energy < 0.005`

**3. merge_alignments** (`aligner/nonverbal_detector.py`)
- Critical fusion logic that resolves temporal conflicts between phonemes and events
- Priority rule: **events override phonemes** on overlap
- Implementation: sorts all segments by start time, then truncates overlapping phonemes to make room for events

**4. TextGridExporter** (`aligner/textgrid_exporter.py`)
- Creates 4-tier TextGrid: Event → Word → Phoneme-CN → IPA-EN
- Chinese-specific: splits pinyin into initials/finals (e.g., "zhun" → "zh" + "un")
- Time allocation: initial gets 30% of unit duration, final gets 70%
- IPA conversion via lookup tables for both Chinese and English

### Data Flow

```
audio.wav + transcript
    ↓
PhonemeAligner (MMS CTC) ──┐
                           ├──→ merge_alignments ──→ TextGridExporter ──→ .TextGrid
detector.detect() ──────────┘      (conflict resolution)
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
