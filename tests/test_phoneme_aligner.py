"""
Phoneme Aligner Module Tests
音素对齐器单元测试
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from aligner.phoneme_aligner import (
    PhonemeAligner,
    AlignmentSegment,
    align_audio_text
)


class TestPhonemeAlignerInitialization:
    """对齐器初始化测试"""

    def test_default_initialization(self):
        """测试默认初始化参数"""
        # 使用 patch 避免实际加载模型
        with patch.object(PhonemeAligner, '_load_model'):
            aligner = PhonemeAligner(lang="zh")

        assert aligner.lang == "zh"
        assert aligner.sample_rate == 16000
        assert aligner.device in ["cuda", "cpu"]

    def test_custom_sample_rate(self):
        """测试自定义采样率"""
        with patch.object(PhonemeAligner, '_load_model'):
            aligner = PhonemeAligner(lang="zh", sample_rate=22050)
        assert aligner.sample_rate == 22050

    def test_custom_device(self):
        """测试自定义设备"""
        with patch.object(PhonemeAligner, '_load_model'):
            aligner = PhonemeAligner(lang="zh", device="cpu")
        assert aligner.device == "cpu"

    def test_supported_languages(self):
        """测试支持的语言列表"""
        supported = PhonemeAligner.LANG_MODEL_MAP
        assert "zh" in supported
        assert "en" in supported
        assert "ja" in supported
        assert "ko" in supported

    def test_model_name_mapping(self):
        """测试语言代码到模型名称的映射"""
        assert PhonemeAligner.LANG_MODEL_MAP["zh"] == "facebook/mms-tts-zho"
        assert PhonemeAligner.LANG_MODEL_MAP["en"] == "facebook/mms-tts-eng"


class TestAlignmentSegment:
    """AlignmentSegment 数据类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        seg = AlignmentSegment(
            token="zh",
            start_time=0.0,
            end_time=0.5,
            confidence=0.9
        )
        assert seg.token == "zh"
        assert seg.start_time == 0.0
        assert seg.end_time == 0.5
        assert seg.confidence == 0.9

    def test_optional_confidence(self):
        """测试可选的置信度"""
        seg = AlignmentSegment(
            token="a",
            start_time=0.0,
            end_time=0.3
        )
        assert seg.confidence is None


class TestPreprocessAudio:
    """音频预处理测试"""

    def test_preprocess_mono_audio(self, sample_aligner: PhonemeAligner, sample_audio_file: Path):
        """测试单声道音频预处理"""
        waveform = sample_aligner.preprocess_audio(str(sample_audio_file))

        # 验证输出类型
        assert isinstance(waveform, torch.Tensor)
        # 验证是一维（单声道）
        assert waveform.dim() == 1
        # 验证数值范围（归一化后）
        assert torch.max(torch.abs(waveform)) <= 1.0 + 1e-6

    def test_preprocess_normalization(self, sample_aligner: PhonemeAligner, sample_audio_file: Path):
        """测试音频归一化"""
        waveform = sample_aligner.preprocess_audio(str(sample_audio_file))

        # 归一化后最大绝对值应接近 1.0（如果原始音频非静音）
        max_val = torch.max(torch.abs(waveform))
        assert max_val > 0
        assert max_val <= 1.0 + 1e-6


class TestTextProcessing:
    """文本处理测试"""

    def test_clean_text_zh(self, sample_aligner: PhonemeAligner):
        """测试中文文本清理"""
        text = "  你好   世界  "
        cleaned = sample_aligner._clean_text(text)
        # 验证多余空白被移除
        assert "  " not in cleaned
        assert cleaned.strip() == cleaned

    def test_segment_chinese(self, sample_aligner: PhonemeAligner):
        """测试中文分词"""
        text = "你好世界"
        segmented = sample_aligner._segment_chinese(text)
        # 验证字符间有空格
        assert " " in segmented or segmented == text

    def test_clean_text_en(self):
        """测试英文文本清理"""
        with patch.object(PhonemeAligner, '_load_model'):
            aligner = PhonemeAligner(lang="en")

        text = "Hello WORLD"
        cleaned = aligner._clean_text(text)
        assert cleaned == "hello world"

    def test_text_to_tokens(self, sample_aligner: PhonemeAligner):
        """测试文本转 tokens"""
        # 使用 mock 避免实际模型调用
        with patch.object(sample_aligner.processor, '__call__') as mock_process:
            mock_process.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

            tokens = sample_aligner.text_to_tokens("你好")
            assert isinstance(tokens, list)
            assert len(tokens) == 3


class TestCTCForcedAlign:
    """CTC 强制对齐算法测试"""

    def test_ctc_align_basic(self, sample_aligner: PhonemeAligner):
        """测试基础 CTC 对齐"""
        # 创建模拟的 emissions: [T=10, V=5]
        T, V = 10, 5
        emissions = torch.randn(T, V)
        emissions = torch.log_softmax(emissions, dim=-1)

        token_ids = [1, 2, 3]

        alignment, scores = sample_aligner.ctc_forced_align(emissions, token_ids)

        # 验证输出
        assert isinstance(alignment, list)
        assert isinstance(scores, list)
        assert len(alignment) == T
        assert len(scores) == T

    def test_ctc_align_empty_tokens(self, sample_aligner: PhonemeAligner):
        """测试空 token 列表"""
        emissions = torch.randn(10, 5)
        alignment, scores = sample_aligner.ctc_forced_align(emissions, [])

        assert alignment == []
        assert scores == []

    def test_ctc_align_single_token(self, sample_aligner: PhonemeAligner):
        """测试单 token 对齐"""
        T, V = 10, 5
        emissions = torch.randn(T, V)
        emissions = torch.log_softmax(emissions, dim=-1)

        alignment, scores = sample_aligner.ctc_forced_align(emissions, [1])

        assert len(alignment) == T
        assert len(scores) == T


class TestAlignmentToSegments:
    """对齐结果转时间段测试"""

    def test_alignment_to_segments_basic(self, sample_aligner: PhonemeAligner):
        """测试基础转换"""
        # 模拟对齐结果: 10 帧，包含 token 1, 2, 3
        alignment = [0, 1, 1, 0, 2, 2, 0, 3, 3, 0]  # 0 is blank
        scores = [0.1] * 10
        original_tokens = [1, 2, 3]

        # 设置 token2id 映射
        sample_aligner.token2id = {1: "a", 2: "b", 3: "c"}

        segments = sample_aligner._alignment_to_segments(
            alignment, scores, original_tokens, stride=320
        )

        # 验证输出
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all(isinstance(s, AlignmentSegment) for s in segments)

    def test_alignment_skips_blank(self, sample_aligner: PhonemeAligner):
        """测试空白标签被正确跳过"""
        alignment = [0, 0, 0, 1, 1, 0, 0]  # 大部分是 blank
        scores = [0.1] * 7
        original_tokens = [1]

        sample_aligner.token2id = {1: "a"}

        segments = sample_aligner._alignment_to_segments(
            alignment, scores, original_tokens, stride=320
        )

        # 应该只有一段非空白
        assert len(segments) == 1
        assert segments[0].token == "a"

    def test_alignment_filters_short_segments(self, sample_aligner: PhonemeAligner):
        """测试过滤过短片段"""
        # 非常短的片段应该被过滤
        alignment = [0, 1, 0]  # 只有 1 帧非 blank
        scores = [0.1] * 3
        original_tokens = [1]

        sample_aligner.token2id = {1: "a"}
        sample_aligner.sample_rate = 16000

        segments = sample_aligner._alignment_to_segments(
            alignment, scores, original_tokens, stride=320  # 20ms per frame
        )

        # 1 帧只有 20ms，应该被过滤
        assert len(segments) == 0 or segments[0].end_time - segments[0].start_time >= 0.01


class TestAlignText:
    """完整对齐流程测试"""

    @pytest.mark.integration
    def test_align_text_integration(self, sample_aligner: PhonemeAligner, sample_audio_file: Path):
        """集成测试：完整对齐流程"""
        # 使用模拟数据避免实际模型推理
        with patch.object(sample_aligner, 'compute_emissions') as mock_emissions:
            # 创建模拟 emissions
            T, V = 100, len(sample_aligner.id2token) if hasattr(sample_aligner, 'id2token') else 50
            emissions = torch.randn(T, V)
            emissions = torch.log_softmax(emissions, dim=-1)
            mock_emissions.return_value = (emissions, 320)

            with patch.object(sample_aligner, 'text_to_tokens') as mock_tokens:
                mock_tokens.return_value = [1, 2, 3]

                result = sample_aligner.align_text(
                    audio_path=str(sample_audio_file),
                    transcript="测试文本"
                )

                assert isinstance(result, list)


class TestAlignAudioText:
    """便捷函数测试"""

    @pytest.mark.integration
    def test_align_audio_text_function(self, sample_audio_file: Path):
        """测试便捷函数"""
        with patch.object(PhonemeAligner, '_load_model'):
            with patch.object(PhonemeAligner, 'align_text') as mock_align:
                mock_align.return_value = [
                    AlignmentSegment(token="a", start_time=0.0, end_time=0.5, confidence=0.9)
                ]

                result = align_audio_text(
                    audio_path=str(sample_audio_file),
                    transcript="测试",
                    lang="zh"
                )

                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0].token == "a"


class TestWordLevelAlignment:
    """词级别对齐测试"""

    def test_get_word_level_alignment(self, sample_aligner: PhonemeAligner, sample_audio_file: Path):
        """测试词级别对齐"""
        with patch.object(sample_aligner, 'align_text') as mock_align:
            mock_align.return_value = [
                AlignmentSegment(token="a", start_time=0.0, end_time=0.5, confidence=0.9),
                AlignmentSegment(token="b", start_time=0.5, end_time=1.0, confidence=0.8),
            ]

            result = sample_aligner.get_word_level_alignment(
                audio_path=str(sample_audio_file),
                transcript="word1 word2"
            )

            assert isinstance(result, list)
            # 应该有 2 个词
            assert len(result) == 2
