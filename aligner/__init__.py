"""
ParaSync Aligner Module
语音对齐核心模块
"""

from .phoneme_aligner import PhonemeAligner, AlignmentSegment, align_audio_text
from .nonverbal_detector import NonverbalEventDetector, NonverbalEvent
from .textgrid_exporter import TextGridExporter

__version__ = "0.1.0"
__all__ = [
    "PhonemeAligner",
    "AlignmentSegment",
    "align_audio_text",
    "NonverbalEventDetector",
    "NonverbalEvent",
    "TextGridExporter",
]
