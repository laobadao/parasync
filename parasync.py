#!/usr/bin/env python3
"""
ParaSync - 语音对齐工具包主入口
支持多层级音素对齐与非语言事件检测

Usage:
    python parasync.py align --audio audio.wav --text "你好世界" --lang zh --output result.TextGrid
    python parasync.py detect --audio audio.wav --output events.json
    python parasync.py pipeline --audio audio.wav --text "你好世界" --output result.TextGrid
"""

import argparse
import sys
import json
from pathlib import Path

from aligner import (
    PhonemeAligner,
    NonverbalEventDetector,
    TextGridExporter,
    align_audio_text,
)
from aligner.nonverbal_detector import merge_alignments, EventType


def cmd_align(args):
    """执行强制对齐"""
    print(f"[ParaSync] 开始对齐: {args.audio}")
    print(f"  语言: {args.lang}")
    print(f"  文本: {args.text}")

    aligner = PhonemeAligner(lang=args.lang)
    segments = aligner.align_text(args.audio, args.text)

    print(f"\n[ParaSync] 对齐完成，共 {len(segments)} 个片段:")
    for seg in segments[:10]:  # 只显示前10个
        print(f"  {seg.token:10} | {seg.start_time:.3f}s - {seg.end_time:.3f}s | {seg.confidence:.3f}")

    if len(segments) > 10:
        print(f"  ... 还有 {len(segments) - 10} 个片段")

    # 保存结果
    if args.output:
        results = [
            {
                "token": seg.token,
                "start": seg.start_time,
                "end": seg.end_time,
                "confidence": seg.confidence,
            }
            for seg in segments
        ]

        if args.output.endswith('.json'):
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif args.output.endswith('.TextGrid'):
            exporter = TextGridExporter()
            duration = segments[-1].end_time if segments else 0
            exporter.export_from_alignment(
                [{"type": "phoneme", **r} for r in results],
                args.output,
                duration
            )

        print(f"\n[ParaSync] 结果已保存: {args.output}")

    return segments


def cmd_detect(args):
    """执行非语言事件检测"""
    print(f"[ParaSync] 开始检测非语言事件: {args.audio}")

    detector = NonverbalEventDetector(
        method=args.method,
        sample_rate=args.sample_rate
    )

    # 指定要检测的事件类型
    events_to_detect = None
    if args.events:
        events_to_detect = [EventType(e) for e in args.events.split(',')]

    events = detector.detect(args.audio, events_to_detect)

    print(f"\n[ParaSync] 检测到 {len(events)} 个事件:")
    for evt in events:
        print(f"  [{evt.event_type.value:8}] | {evt.start_time:.3f}s - {evt.end_time:.3f}s | {evt.confidence:.3f}")

    # 保存结果
    if args.output:
        results = [
            {
                "type": evt.event_type.value,
                "start": evt.start_time,
                "end": evt.end_time,
                "confidence": evt.confidence,
            }
            for evt in events
        ]

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[ParaSync] 结果已保存: {args.output}")

    return events


def cmd_pipeline(args):
    """执行完整流程：对齐 + 事件检测 + 融合 + 导出"""
    print("=" * 60)
    print("[ParaSync] 完整对齐流程")
    print("=" * 60)

    # 步骤 1: 音素对齐
    print("\n📍 步骤 1/3: 音素对齐")
    aligner = PhonemeAligner(lang=args.lang)
    phoneme_segments = aligner.align_text(args.audio, args.text)
    print(f"   ✓ 对齐完成: {len(phoneme_segments)} 个音素段")

    # 步骤 2: 非语言事件检测
    print("\n📍 步骤 2/3: 非语言事件检测")
    detector = NonverbalEventDetector(method=args.method)
    events = detector.detect(args.audio)
    print(f"   ✓ 检测完成: {len(events)} 个事件")

    # 步骤 3: 融合与导出
    print("\n📍 步骤 3/3: 融合对齐结果并导出")

    # 融合
    merged = merge_alignments(phoneme_segments, events)
    print(f"   ✓ 融合完成: {len(merged)} 个片段")

    # 计算总时长
    duration = max(seg["end"] for seg in merged) if merged else 0

    # 导出 TextGrid
    exporter = TextGridExporter()

    # 准备各层级数据
    events_data = [m for m in merged if m["type"] == "event"]
    phonemes_data = [
        {
            "pinyin": m["token"],
            "start": m["start"],
            "end": m["end"]
        }
        for m in merged if m["type"] == "phoneme"
    ]

    # 简单的词层级（从文本分割）
    words = args.text.split()
    word_duration = duration / len(words) if words else 0
    words_data = [
        {
            "word": w,
            "start": i * word_duration,
            "end": (i + 1) * word_duration
        }
        for i, w in enumerate(words)
    ]

    exporter.export_multi_tier(
        events=events_data,
        words=words_data,
        phonemes_cn=phonemes_data,
        phonemes_ipa=[{"token": p["token"], "start": p["start"], "end": p["end"]}
                      for p in phonemes_data],
        output_path=args.output,
        duration=duration
    )

    print("\n" + "=" * 60)
    print("[ParaSync] 处理完成!")
    print(f"   输出文件: {args.output}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ParaSync - 语音对齐工具包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础对齐
  python parasync.py align -a audio.wav -t "你好世界" -l zh -o result.json

  # 事件检测
  python parasync.py detect -a audio.wav -o events.json

  # 完整流程
  python parasync.py pipeline -a audio.wav -t "你好世界" -o result.TextGrid
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # align 命令
    align_parser = subparsers.add_parser('align', help='执行强制对齐')
    align_parser.add_argument('-a', '--audio', required=True, help='音频文件路径')
    align_parser.add_argument('-t', '--text', required=True, help='参考文本')
    align_parser.add_argument('-l', '--lang', default='zh', help='语言代码 (zh/en/...)')
    align_parser.add_argument('-o', '--output', help='输出文件路径 (.json 或 .TextGrid)')
    align_parser.set_defaults(func=cmd_align)

    # detect 命令
    detect_parser = subparsers.add_parser('detect', help='检测非语言事件')
    detect_parser.add_argument('-a', '--audio', required=True, help='音频文件路径')
    detect_parser.add_argument('-m', '--method', default='heuristic',
                               choices=['heuristic', 'model'], help='检测方法')
    detect_parser.add_argument('-e', '--events', help='要检测的事件类型，逗号分隔 (hx,laugh,cry)')
    detect_parser.add_argument('-sr', '--sample-rate', type=int, default=16000, help='采样率')
    detect_parser.add_argument('-o', '--output', help='输出文件路径 (.json)')
    detect_parser.set_defaults(func=cmd_detect)

    # pipeline 命令
    pipeline_parser = subparsers.add_parser('pipeline', help='执行完整流程')
    pipeline_parser.add_argument('-a', '--audio', required=True, help='音频文件路径')
    pipeline_parser.add_argument('-t', '--text', required=True, help='参考文本')
    pipeline_parser.add_argument('-l', '--lang', default='zh', help='语言代码')
    pipeline_parser.add_argument('-m', '--method', default='heuristic', help='事件检测方法')
    pipeline_parser.add_argument('-o', '--output', default='result.TextGrid', help='输出文件路径')
    pipeline_parser.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
