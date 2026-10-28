"""generate_random_combos.py

随机生成 N 个 DA combo（与 best_combo.json 相同格式），用于对比实验。

示例：
  python generate_random_combos.py --n 5 --seed 123 --out_dir ga_results/Random_DA

输出：
  ga_results/Random_DA/random_combo_01.json
  ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from util import combo_to_json, generate_random_combos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="生成 combo 数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument("--length", type=int, default=3, help="combo 长度（默认 3 对应 R/G/B）")
    parser.add_argument("--out_dir", type=str, default="ga_results/Random_DA", help="输出目录")
    parser.add_argument(
        "--prefix",
        type=str,
        default="random_combo_",
        help="输出文件名前缀（将生成 prefix01.json ...）",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="以缩进 JSON 保存（更易读）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = generate_random_combos(n=args.n, seed=args.seed, length=args.length, ensure_unique=True)

    for i, combo in enumerate(combos, start=1):
        methods = [g[0] for g in combo]
        raw_json = combo_to_json(combo)
        if args.pretty:
            content = json.dumps(json.loads(raw_json), indent=2, ensure_ascii=False)
        else:
            content = raw_json

        out_path = out_dir / f"{args.prefix}{i:02d}.json"
        out_path.write_text(content, encoding="utf-8")
        print(f"[{i:02d}] {methods} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
