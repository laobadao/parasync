"""
ParaSync Package
语音对齐工具包
"""

__version__ = "0.1.0"

from aligner.phoneme_aligner import PhonemeAligner, AlignmentSegment, align_audio_text
from aligner.nonverbal_detector import (
    NonverbalEventDetector,
    NonverbalEvent,
    EventType,
    merge_alignments,
)
from aligner.textgrid_exporter import TextGridExporter
from parasync.tts_generator import ChatTTSGenerator, TTSUtterance, generate_test_audio

__all__ = [
    # Aligner
    "PhonemeAligner",
    "AlignmentSegment",
    "align_audio_text",
    # Detector
    "NonverbalEventDetector",
    "NonverbalEvent",
    "EventType",
    "merge_alignments",
    # Exporter
    "TextGridExporter",
    # TTS
    "ChatTTSGenerator",
    "TTSUtterance",
    "generate_test_audio",
]
