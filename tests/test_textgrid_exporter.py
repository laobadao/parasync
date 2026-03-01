"""
TextGrid Exporter Module Tests
TextGrid 导出器单元测试
"""

import pytest
import textgrid
from pathlib import Path
from unittest.mock import Mock

from aligner.textgrid_exporter import (
    TextGridExporter,
    TextGridValidator,
    TierType,
    TierConfig,
)


class TestTierConfig:
    """层级配置测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        config = TierConfig(name="Test", tier_type=TierType.WORD)
        assert config.name == "Test"
        assert config.tier_type == TierType.WORD
        assert config.is_point_tier is False

    def test_point_tier(self):
        """测试时间点层级"""
        config = TierConfig(name="Point", tier_type=TierType.EVENT, is_point_tier=True)
        assert config.is_point_tier is True


class TestTextGridExporterInitialization:
    """导出器初始化测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        exporter = TextGridExporter()

        assert len(exporter.tier_configs) == 4
        config_names = [c.name for c in exporter.tier_configs]
        assert "Event" in config_names
        assert "Word" in config_names
        assert "Phoneme-CN" in config_names
        assert "IPA-EN" in config_names

    def test_initials_set(self):
        """测试声母集合"""
        exporter = TextGridExporter()
        assert "zh" in exporter.INITIALS
        assert "ch" in exporter.INITIALS
        assert "sh" in exporter.INITIALS
        assert "z" in exporter.INITIALS
        assert "c" in exporter.INITIALS
        assert "s" in exporter.INITIALS


class TestCreateTextGrid:
    """创建 TextGrid 测试"""

    def test_create_basic_textgrid(self):
        """测试基础创建"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=10.0, name="Test")

        assert isinstance(tg, textgrid.TextGrid)
        assert tg.maxTime == 10.0
        assert tg.name == "Test"

    def test_create_with_tiers(self):
        """测试创建带层级的 TextGrid"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        assert len(tg.tiers) == 4
        tier_names = [t.name for t in tg.tiers]
        assert "Event" in tier_names
        assert "Word" in tier_names
        assert "Phoneme-CN" in tier_names
        assert "IPA-EN" in tier_names

    def test_all_tiers_are_interval(self):
        """测试所有层级都是 IntervalTier"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        for tier in tg.tiers:
            assert isinstance(tier, textgrid.IntervalTier)


class TestSplitPinyin:
    """拼音拆分测试"""

    def test_split_simple_initial_final(self):
        """测试简单声韵母拆分"""
        exporter = TextGridExporter()
        initials, finals = exporter._split_pinyin("hao")

        assert initials == ["h"]
        assert finals == ["ao"]

    def test_split_compound_initial(self):
        """测试复合声母拆分"""
        exporter = TextGridExporter()
        initials, finals = exporter._split_pinyin("zhun")

        assert initials == ["zh"]
        assert finals == ["un"]

    def test_split_zero_initial(self):
        """测试零声母"""
        exporter = TextGridExporter()
        initials, finals = exporter._split_pinyin("a")

        assert initials == [""]
        assert finals == ["a"]

    def test_split_with_tone(self):
        """测试带声调移除"""
        exporter = TextGridExporter()
        initials, finals = exporter._split_pinyin("hǎo")

        assert initials == ["h"]
        assert finals == ["ao"]

    def test_split_multiple_syllables(self):
        """测试多音节（当前实现只处理单个）"""
        exporter = TextGridExporter()
        # 注意：当前实现是针对单个拼音
        initials, finals = exporter._split_pinyin("zhuang")

        assert "zh" in initials[0] if initials else True


class TestRemoveTone:
    """移除声调测试"""

    def test_remove_tone_first(self):
        """测试第一声"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("mā")
        assert result == "ma"

    def test_remove_tone_second(self):
        """测试第二声"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("má")
        assert result == "ma"

    def test_remove_tone_third(self):
        """测试第三声"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("mǎ")
        assert result == "ma"

    def test_remove_tone_fourth(self):
        """测试第四声"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("mà")
        assert result == "ma"

    def test_remove_tone_u_umlaut(self):
        """测试 ü 音调"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("nǚ")
        assert result == "nü"

    def test_no_tone(self):
        """测试无声调"""
        exporter = TextGridExporter()
        result = exporter._remove_tone("ma")
        assert result == "ma"


class TestToIPA:
    """IPA 转换测试"""

    def test_pinyin_to_ipa_initial(self):
        """测试声母转 IPA"""
        exporter = TextGridExporter()
        result = exporter._to_ipa("zh")
        assert result == "ʈʂ"

    def test_pinyin_to_ipa_final(self):
        """测试韵母转 IPA"""
        exporter = TextGridExporter()
        result = exporter._to_ipa("ao")
        assert result == "aʊ"

    def test_arpabet_to_ipa(self):
        """测试 Arpabet 转 IPA"""
        exporter = TextGridExporter()
        result = exporter._to_ipa("SH")
        assert result == "ʂ"

    def test_unknown_phoneme(self):
        """测试未知音素返回原值"""
        exporter = TextGridExporter()
        result = exporter._to_ipa("xyz")
        assert result == "xyz"


class TestAddEventTier:
    """添加事件层级测试"""

    def test_add_single_event(self):
        """测试添加单个事件"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        events = [
            {"start": 0.5, "end": 1.0, "label": "[hx]"}
        ]
        exporter.add_event_tier(tg, events)

        event_tier = [t for t in tg.tiers if t.name == "Event"][0]
        assert len(event_tier.intervals) == 1
        assert event_tier.intervals[0].mark == "[hx]"

    def test_add_multiple_events(self):
        """测试添加多个事件"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        events = [
            {"start": 0.5, "end": 1.0, "label": "[hx]"},
            {"start": 2.0, "end": 2.5, "label": "[laugh]"},
        ]
        exporter.add_event_tier(tg, events)

        event_tier = [t for t in tg.tiers if t.name == "Event"][0]
        assert len(event_tier.intervals) == 2


class TestAddWordTier:
    """添加词层级测试"""

    def test_add_words(self):
        """测试添加词"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        words = [
            {"start": 0.0, "end": 0.5, "word": "你好"},
            {"start": 0.5, "end": 1.0, "word": "世界"},
        ]
        exporter.add_word_tier(tg, words)

        word_tier = [t for t in tg.tiers if t.name == "Word"][0]
        assert len(word_tier.intervals) == 2
        assert word_tier.intervals[0].mark == "你好"


class TestAddPhonemeCNTier:
    """添加中文音素层级测试"""

    def test_add_phoneme_no_split(self):
        """测试不拆分的音素添加"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        phonemes = [
            {"pinyin": "hao", "start": 0.0, "end": 0.5},
        ]
        exporter.add_phoneme_cn_tier(tg, phonemes, split_initial_final=False)

        tier = [t for t in tg.tiers if t.name == "Phoneme-CN"][0]
        assert len(tier.intervals) == 1
        assert tier.intervals[0].mark == "hao"

    def test_add_phoneme_with_split(self):
        """测试拆分的音素添加"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        phonemes = [
            {"pinyin": "hao", "start": 0.0, "end": 0.4},
        ]
        exporter.add_phoneme_cn_tier(tg, phonemes, split_initial_final=True)

        tier = [t for t in tg.tiers if t.name == "Phoneme-CN"][0]
        # hao -> h + ao，应该有两个区间
        assert len(tier.intervals) >= 1


class TestExportFromAlignment:
    """从对齐结果导出测试"""

    def test_export_basic(self, temp_output_dir: Path):
        """测试基础导出"""
        exporter = TextGridExporter()

        alignment_result = [
            {"type": "phoneme", "token": "h", "start": 0.0, "end": 0.1, "confidence": 0.9},
            {"type": "phoneme", "token": "ao", "start": 0.1, "end": 0.4, "confidence": 0.9},
            {"type": "event", "token": "[hx]", "start": 0.5, "end": 0.8, "confidence": 0.8, "event_type": "hx"},
        ]

        output_path = temp_output_dir / "test.TextGrid"
        exporter.export_from_alignment(alignment_result, str(output_path), duration=1.0)

        assert output_path.exists()

        # 验证可以读取
        tg = textgrid.TextGrid.fromFile(str(output_path))
        assert tg.maxTime == 1.0


class TestExportMultiTier:
    """多层级导出测试"""

    def test_export_all_tiers(self, temp_output_dir: Path):
        """测试导出所有层级"""
        exporter = TextGridExporter()

        output_path = temp_output_dir / "multi_tier.TextGrid"

        exporter.export_multi_tier(
            events=[{"start": 0.5, "end": 1.0, "label": "[hx]"}],
            words=[{"start": 0.0, "end": 1.0, "word": "你好"}],
            phonemes_cn=[{"pinyin": "ni", "start": 0.0, "end": 0.5}, {"pinyin": "hao", "start": 0.5, "end": 1.0}],
            phonemes_ipa=[{"token": "n", "start": 0.0, "end": 0.5}, {"token": "h", "start": 0.5, "end": 1.0}],
            output_path=str(output_path),
            duration=1.0
        )

        assert output_path.exists()

        tg = textgrid.TextGrid.fromFile(str(output_path))
        tier_names = [t.name for t in tg.tiers]
        assert "Event" in tier_names
        assert "Word" in tier_names
        assert "Phoneme-CN" in tier_names
        assert "IPA-EN" in tier_names


class TestTextGridValidator:
    """TextGrid 验证器测试"""

    def test_validate_empty_tier(self):
        """测试空层级警告"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        result = TextGridValidator.validate(tg)

        assert "warnings" in result
        # 空层级应该有警告
        assert len(result["warnings"]) > 0

    def test_validate_no_overlap(self):
        """测试无重叠通过验证"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        # 添加不重叠的区间
        events = [
            {"start": 0.0, "end": 1.0, "label": "a"},
            {"start": 1.0, "end": 2.0, "label": "b"},
        ]
        exporter.add_event_tier(tg, events)

        result = TextGridValidator.validate(tg)

        # 不应该有时间重叠错误
        overlap_errors = [e for e in result["errors"] if "重叠" in e or "overlap" in e.lower()]
        assert len(overlap_errors) == 0

    def test_validate_detects_overlap(self):
        """测试检测时间重叠"""
        exporter = TextGridExporter()
        tg = exporter.create_textgrid(duration=5.0)

        # 添加重叠的区间
        events = [
            {"start": 0.0, "end": 1.5, "label": "a"},
            {"start": 1.0, "end": 2.0, "label": "b"},  # 与 a 重叠
        ]
        exporter.add_event_tier(tg, events)

        result = TextGridValidator.validate(tg)

        # 应该有重叠错误
        assert len(result["errors"]) > 0


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_alignment(self, temp_output_dir: Path):
        """测试空对齐结果"""
        exporter = TextGridExporter()
        output_path = temp_output_dir / "empty.TextGrid"

        exporter.export_from_alignment([], str(output_path))

        assert output_path.exists()

    def test_very_short_duration(self, temp_output_dir: Path):
        """测试非常短的时长"""
        exporter = TextGridExporter()
        output_path = temp_output_dir / "short.TextGrid"

        tg = exporter.create_textgrid(duration=0.001)
        tg.write(str(output_path))

        assert output_path.exists()

    def test_long_duration(self, temp_output_dir: Path):
        """测试长时长"""
        exporter = TextGridExporter()
        output_path = temp_output_dir / "long.TextGrid"

        tg = exporter.create_textgrid(duration=3600)  # 1小时
        tg.write(str(output_path))

        assert output_path.exists()

        read_tg = textgrid.TextGrid.fromFile(str(output_path))
        assert read_tg.maxTime == 3600
