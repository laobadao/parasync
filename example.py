#!/usr/bin/env python3
"""
ParaSync 使用示例
展示如何进行音素对齐、事件检测和 TextGrid 导出
"""

from aligner import (
    PhonemeAligner,
    NonverbalEventDetector,
    TextGridExporter,
)
from aligner.nonverbal_detector import merge_alignments


def example_basic_alignment():
    """示例 1: 基础音素对齐"""
    print("=" * 60)
    print("示例 1: 基础音素对齐")
    print("=" * 60)

    # 初始化对齐器
    aligner = PhonemeAligner(lang="zh")

    # 对齐音频（假设文件存在）
    # segments = aligner.align_text("audio.wav", "你好世界，我是 ParaSync")
    #
    # for seg in segments:
    #     print(f"{seg.token}: {seg.start_time:.3f}s - {seg.end_time:.3f}s")

    print("✓ 请替换为实际音频文件路径进行测试")


def example_event_detection():
    """示例 2: 非语言事件检测"""
    print("\n" + "=" * 60)
    print("示例 2: 非语言事件检测")
    print("=" * 60)

    # 初始化检测器
    detector = NonverbalEventDetector(method="heuristic")

    # 检测事件
    # events = detector.detect("audio_with_breath.wav")
    #
    # for evt in events:
    #     print(f"[{evt.event_type.value}]: {evt.start_time:.3f}s - {evt.end_time:.3f}s")

    print("✓ 请替换为实际音频文件路径进行测试")


def example_full_pipeline():
    """示例 3: 完整流程 - 对齐、检测、融合、导出"""
    print("\n" + "=" * 60)
    print("示例 3: 完整流程")
    print("=" * 60)

    # 假设的音频和文本
    audio_path = "sample.wav"
    transcript = "你好，[hx] 我是 ParaSync"

    print(f"音频: {audio_path}")
    print(f"文本: {transcript}")
    print("\n处理流程:")

    # 1. 音素对齐
    print("  1. 音素对齐...")
    aligner = PhonemeAligner(lang="zh")
    # phoneme_segments = aligner.align_text(audio_path, transcript)
    print("     ✓ 完成")

    # 2. 非语言事件检测
    print("  2. 非语言事件检测...")
    detector = NonverbalEventDetector()
    # events = detector.detect(audio_path)
    print("     ✓ 完成")

    # 3. 融合对齐结果
    print("  3. 融合对齐结果...")
    # merged = merge_alignments(phoneme_segments, events)
    print("     ✓ 完成")

    # 4. 导出 TextGrid
    print("  4. 导出 TextGrid...")
    exporter = TextGridExporter()
    # exporter.export_from_alignment(merged, "result.TextGrid")
    print("     ✓ 完成")

    print("\n输出文件: result.TextGrid")


def example_textgrid_structure():
    """示例 4: 创建多层级 TextGrid"""
    print("\n" + "=" * 60)
    print("示例 4: 多层级 TextGrid 结构")
    print("=" * 60)

    exporter = TextGridExporter()

    # 创建示例数据
    duration = 3.0

    events = [
        {"start": 0.5, "end": 0.8, "label": "[hx]"},
        {"start": 2.0, "end": 2.3, "label": "[laugh]"},
    ]

    words = [
        {"start": 0.0, "end": 0.5, "word": "你好"},
        {"start": 0.8, "end": 1.5, "word": "世界"},
        {"start": 1.5, "end": 2.0, "word": "我是"},
        {"start": 2.3, "end": 3.0, "word": "ParaSync"},
    ]

    phonemes_cn = [
        {"pinyin": "ni", "start": 0.0, "end": 0.25},
        {"pinyin": "hao", "start": 0.25, "end": 0.5},
        {"pinyin": "shi", "start": 0.8, "end": 1.0},
        {"pinyin": "jie", "start": 1.0, "end": 1.5},
    ]

    phonemes_ipa = [
        {"token": "n", "start": 0.0, "end": 0.25},
        {"token": "x", "start": 0.25, "end": 0.5},
        {"token": "ʂ", "start": 0.8, "end": 1.0},
        {"token": "tɕ", "start": 1.0, "end": 1.5},
    ]

    # 导出
    exporter.export_multi_tier(
        events=events,
        words=words,
        phonemes_cn=phonemes_cn,
        phonemes_ipa=phonemes_ipa,
        output_path="example.TextGrid",
        duration=duration
    )

    print("✓ 示例 TextGrid 已创建: example.TextGrid")
    print("\n层级结构:")
    print("  - Event: 非语言事件 [hx], [laugh]")
    print("  - Word: 汉字/单词")
    print("  - Phoneme-CN: 拼音声韵母拆分")
    print("  - IPA-EN: 国际音标")


def example_pinyin_split():
    """示例 5: 拼音声韵母拆分"""
    print("\n" + "=" * 60)
    print("示例 5: 拼音声韵母拆分")
    print("=" * 60)

    exporter = TextGridExporter()

    test_cases = [
        "zhun",
        "bei",
        "hao",
        "zhuang",
        "chuan",
        "a",
        "er",
    ]

    print("拼音拆分示例:")
    for pinyin in test_cases:
        initials, finals = exporter._split_pinyin(pinyin)
        print(f"  {pinyin:8} -> 声母: {initials[0] if initials else '':4}, 韵母: {finals[0] if finals else ''}")


if __name__ == "__main__":
    print("\n" + "🎙️  ParaSync - 语音对齐工具包示例\n")

    # 运行所有示例
    example_basic_alignment()
    example_event_detection()
    example_full_pipeline()
    example_textgrid_structure()
    example_pinyin_split()

    print("\n" + "=" * 60)
    print("所有示例展示完成!")
    print("=" * 60)
