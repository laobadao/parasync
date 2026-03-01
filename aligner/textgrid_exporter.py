"""
ParaSync TextGrid Exporter Module
TextGrid 高级导出模块，支持多层级标注
"""

import textgrid
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
import json


class TierType(Enum):
    """TextGrid 层级类型"""
    EVENT = "Event"
    WORD = "Word"
    PHONEME_CN = "Phoneme-CN"
    IPA_EN = "IPA-EN"


@dataclass
class TierConfig:
    """层级配置"""
    name: str
    tier_type: TierType
    is_point_tier: bool = False


class TextGridExporter:
    """
    TextGrid 导出器
    支持四个标准层级：Event, Word, Phoneme-CN, IPA-EN
    """

    # 中文声韵母映射表
    INITIALS = {
        'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
        'g', 'k', 'h', 'j', 'q', 'x',
        'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w'
    }

    # 拼音到 IPA 的映射（简化版）
    PINYIN_TO_IPA = {
        # 声母
        'b': 'p', 'p': 'pʰ', 'm': 'm', 'f': 'f',
        'd': 't', 't': 'tʰ', 'n': 'n', 'l': 'l',
        'g': 'k', 'k': 'kʰ', 'h': 'x',
        'j': 'tɕ', 'q': 'tɕʰ', 'x': 'ɕ',
        'zh': 'ʈʂ', 'ch': 'ʈʂʰ', 'sh': 'ʂ', 'r': 'ʐ',
        'z': 'ts', 'c': 'tsʰ', 's': 's',
        'y': 'j', 'w': 'w',
        # 常见韵母（简化）
        'a': 'a', 'o': 'o', 'e': 'ɤ', 'i': 'i', 'u': 'u', 'ü': 'y',
        'ai': 'aɪ', 'ei': 'eɪ', 'ao': 'aʊ', 'ou': 'oʊ',
        'an': 'an', 'en': 'ən', 'ang': 'aŋ', 'eng': 'əŋ',
        'ong': 'ʊŋ', 'er': 'ɚ',
        'i': 'i', 'ia': 'ja', 'iao': 'jaʊ', 'ian': 'jɛn',
        'in': 'in', 'iang': 'jaŋ', 'ing': 'iŋ', 'iong': 'jʊŋ',
        'u': 'u', 'ua': 'wa', 'uo': 'wo', 'uai': 'waɪ',
        'ui': 'weɪ', 'uan': 'wan', 'un': 'wən', 'uang': 'waŋ',
        'ü': 'y', 'üe': 'ɥɛ', 'ün': 'yn',
    }

    def __init__(self):
        """初始化导出器"""
        self.tier_configs = [
            TierConfig("Event", TierType.EVENT),
            TierConfig("Word", TierType.WORD),
            TierConfig("Phoneme-CN", TierType.PHONEME_CN),
            TierConfig("IPA-EN", TierType.IPA_EN),
        ]

    def create_textgrid(
        self,
        duration: float,
        name: str = "Alignment"
    ) -> textgrid.TextGrid:
        """
        创建空的 TextGrid 结构

        Args:
            duration: 音频总时长
            name: TextGrid 名称

        Returns:
            TextGrid 对象
        """
        tg = textgrid.TextGrid(name=name, maxTime=duration)

        for config in self.tier_configs:
            tier = textgrid.IntervalTier(
                name=config.name,
                maxTime=duration
            )
            tg.append(tier)

        return tg

    def add_event_tier(
        self,
        tg: textgrid.TextGrid,
        events: List[Dict],
        tier_name: str = "Event"
    ):
        """
        添加事件层级

        Args:
            tg: TextGrid 对象
            events: 事件列表，格式 [{"start": 0.0, "end": 0.5, "label": "[hx]"}, ...]
            tier_name: 层级名称
        """
        tier = self._get_or_create_tier(tg, tier_name)

        for event in events:
            interval = textgrid.Interval(
                minTime=event["start"],
                maxTime=event["end"],
                mark=event["label"]
            )
            tier.addInterval(interval)

    def add_word_tier(
        self,
        tg: textgrid.TextGrid,
        words: List[Dict],
        tier_name: str = "Word"
    ):
        """
        添加词层级

        Args:
            tg: TextGrid 对象
            words: 词列表，格式 [{"start": 0.0, "end": 0.5, "word": "你好"}, ...]
            tier_name: 层级名称
        """
        tier = self._get_or_create_tier(tg, tier_name)

        for word in words:
            interval = textgrid.Interval(
                minTime=word["start"],
                maxTime=word["end"],
                mark=word["word"]
            )
            tier.addInterval(interval)

    def add_phoneme_cn_tier(
        self,
        tg: textgrid.TextGrid,
        phonemes: List[Dict],
        tier_name: str = "Phoneme-CN",
        split_initial_final: bool = True
    ):
        """
        添加中文音素层级，支持声韵母拆分

        Args:
            tg: TextGrid 对象
            phonemes: 音素列表
            tier_name: 层级名称
            split_initial_final: 是否拆分为声母和韵母
        """
        tier = self._get_or_create_tier(tg, tier_name)

        for phone in phonemes:
            pinyin = phone.get("pinyin", phone.get("token", ""))
            start = phone["start"]
            end = phone["end"]

            if split_initial_final and pinyin:
                initials, finals = self._split_pinyin(pinyin)

                if len(initials) == len(finals):
                    # 计算每个声韵母的时长
                    total_units = len(initials) + len(finals)
                    unit_duration = (end - start) / total_units

                    current_time = start
                    for init, final in zip(initials, finals):
                        # 声母通常较短
                        init_duration = unit_duration * 0.3
                        final_duration = unit_duration * 1.7

                        if init:
                            tier.addInterval(textgrid.Interval(
                                minTime=current_time,
                                maxTime=current_time + init_duration,
                                mark=init
                            ))
                            current_time += init_duration

                        tier.addInterval(textgrid.Interval(
                            minTime=current_time,
                            maxTime=current_time + final_duration,
                            mark=final
                        ))
                        current_time += final_duration
                else:
                    # 无法拆分，直接使用原拼音
                    tier.addInterval(textgrid.Interval(
                        minTime=start,
                        maxTime=end,
                        mark=pinyin
                    ))
            else:
                tier.addInterval(textgrid.Interval(
                    minTime=start,
                    maxTime=end,
                    mark=pinyin
                ))

    def add_ipa_en_tier(
        self,
        tg: textgrid.TextGrid,
        phonemes: List[Dict],
        tier_name: str = "IPA-EN"
    ):
        """
        添加 IPA 音标层级（英文）

        Args:
            tg: TextGrid 对象
            phonemes: 音素列表
            tier_name: 层级名称
        """
        tier = self._get_or_create_tier(tg, tier_name)

        for phone in phonemes:
            # 转换为 IPA
            ipa = self._to_ipa(phone.get("token", ""))

            interval = textgrid.Interval(
                minTime=phone["start"],
                maxTime=phone["end"],
                mark=ipa
            )
            tier.addInterval(interval)

    def _get_or_create_tier(
        self,
        tg: textgrid.TextGrid,
        name: str
    ) -> textgrid.IntervalTier:
        """获取或创建层级"""
        for tier in tg.tiers:
            if tier.name == name:
                return tier

        # 创建新层级
        tier = textgrid.IntervalTier(name=name, maxTime=tg.maxTime)
        tg.append(tier)
        return tier

    def _split_pinyin(self, pinyin: str) -> Tuple[List[str], List[str]]:
        """
        将拼音拆分为声母和韵母

        Args:
            pinyin: 拼音字符串（如 "zhun", "hao", "a"）

        Returns:
            (initials, finals): 声母列表和韵母列表
        """
        # 处理带声调的拼音
        pinyin_clean = self._remove_tone(pinyin)

        # 识别声母
        initial = ""
        final = pinyin_clean

        # 按长度降序尝试匹配声母
        for init in sorted(self.INITIALS, key=len, reverse=True):
            if pinyin_clean.startswith(init):
                initial = init
                final = pinyin_clean[len(init):]
                break

        return [initial] if initial else [""], [final] if final else [""]

    def _remove_tone(self, pinyin: str) -> str:
        """移除声调符号"""
        # 声调映射
        tone_map = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'ü', 'ǘ': 'ü', 'ǚ': 'ü', 'ǜ': 'ü',
        }

        result = ""
        for char in pinyin:
            result += tone_map.get(char, char)

        return result

    def _to_ipa(self, phoneme: str) -> str:
        """将音素转换为 IPA 符号"""
        # 中文拼音转 IPA
        if phoneme in self.PINYIN_TO_IPA:
            return self.PINYIN_TO_IPA[phoneme]

        # 英文音素转 IPA（简化映射）
        arpabet_to_ipa = {
            'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
            'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
            'EH': 'ɛ', 'ER': 'ɚ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
            'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
            'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
            'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
            'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
            'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
        }

        return arpabet_to_ipa.get(phoneme.upper(), phoneme)

    def export_from_alignment(
        self,
        alignment_result: List[Dict],
        output_path: str,
        duration: Optional[float] = None
    ):
        """
        从融合的对齐结果导出 TextGrid

        Args:
            alignment_result: 融合后的对齐结果
            output_path: 输出文件路径
            duration: 音频总时长，None 则自动计算
        """
        # 计算总时长
        if duration is None:
            duration = max(seg["end"] for seg in alignment_result)

        # 创建 TextGrid
        tg = self.create_textgrid(duration)

        # 分离不同类型的段
        events = [s for s in alignment_result if s["type"] == "event"]
        phonemes = [s for s in alignment_result if s["type"] == "phoneme"]

        # 填充各层级
        # 事件使用 "token" 或 "label" 字段
        self.add_event_tier(tg, [
            {"start": e["start"], "end": e["end"], "label": e.get("token", e.get("label", ""))}
            for e in events
        ])

        # 中文音素层级
        self.add_phoneme_cn_tier(tg, [
            {"pinyin": p["token"], "start": p["start"], "end": p["end"]}
            for p in phonemes
        ])

        # 保存
        tg.write(output_path)
        print(f"[ParaSync] TextGrid 已导出: {output_path}")

    def export_multi_tier(
        self,
        events: List[Dict],
        words: List[Dict],
        phonemes_cn: List[Dict],
        phonemes_ipa: List[Dict],
        output_path: str,
        duration: float
    ):
        """
        导出完整的多层级 TextGrid

        Args:
            events: 事件层级
            words: 词层级
            phonemes_cn: 中文音素层级
            phonemes_ipa: IPA 音标层级
            output_path: 输出路径
            duration: 总时长
        """
        tg = self.create_textgrid(duration)

        self.add_event_tier(tg, events)
        self.add_word_tier(tg, words)
        self.add_phoneme_cn_tier(tg, phonemes_cn)
        self.add_ipa_en_tier(tg, phonemes_ipa)

        tg.write(output_path)
        print(f"[ParaSync] 多层级 TextGrid 已导出: {output_path}")


class TextGridValidator:
    """TextGrid 验证器"""

    @staticmethod
    def validate(tg: textgrid.TextGrid) -> Dict[str, List[str]]:
        """
        验证 TextGrid 的完整性和一致性

        Returns:
            {"errors": [...], "warnings": [...]}
        """
        errors = []
        warnings = []

        # 检查空层级
        for tier in tg.tiers:
            if not tier.intervals:
                warnings.append(f"层级 '{tier.name}' 为空")

        # 检查时间对齐
        for tier in tg.tiers:
            if isinstance(tier, textgrid.IntervalTier):
                prev_end = 0
                for interval in tier.intervals:
                    if interval.minTime < prev_end - 0.001:  # 允许 1ms 浮点误差
                        errors.append(
                            f"层级 '{tier.name}' 存在时间重叠: "
                            f"{interval.minTime} < {prev_end}"
                        )
                    prev_end = interval.maxTime

        return {"errors": errors, "warnings": warnings}


if __name__ == "__main__":
    print("ParaSync TextGridExporter 模块加载成功")

    # 简单测试
    exporter = TextGridExporter()

    # 测试拼音拆分
    print("\n拼音拆分测试:")
    test_pinyins = ["zhun", "hao", "a", "zhan", "chuang"]
    for py in test_pinyins:
        init, final = exporter._split_pinyin(py)
        print(f"  {py} -> 声母: {init}, 韵母: {final}")
