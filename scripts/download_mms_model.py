#!/usr/bin/env python3
"""
下载 MMS 模型脚本
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aligner.mms_aligner import MMSAligner


def main():
    print("=" * 60)
    print("📥 ParaSync MMS 模型下载工具")
    print("=" * 60)
    print()

    output_dir = "models/mms"

    try:
        model_path = MMSAligner.download_model(output_dir)
        print()
        print("✅ 下载成功！")
        print(f"   模型路径: {model_path}")
        print()
        print("现在可以使用 MMS 对齐器了:")
        print("  from aligner.mms_aligner import MMSAligner")
        print("  aligner = MMSAligner()")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
