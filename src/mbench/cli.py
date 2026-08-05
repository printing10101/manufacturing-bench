"""CLI：mbench。

子命令：
- mbench diag <results.json>      负 R² 诊断（从结果 JSON 提取 y_true/y_pred，或直接传数组 JSON）
- mbench checklist [--dir D] [--json F]   投稿检查清单
- mbench stats <a.json> <b.json>  两组跨 seed 指标对比（配对 t + 效应量 + 贝叶斯）
- mbench version
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mbench import __version__
from mbench.checklist import SubmissionChecklist
from mbench.diagnostics import NegativeR2Diagnoser
from mbench.stats import hierarchical_diff, mean_ci, paired_t, welch_t


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_series(data: dict, key: str = "y_true") -> list:
    """从结果 JSON 提取数值序列（支持嵌套：metric 列表 or 直接数组）。"""
    if key in data:
        v = data[key]
        return v if isinstance(v, list) else [v]
    # 递归查找
    for v in data.values():
        if isinstance(v, dict):
            r = _extract_series(v, key)
            if r:
                return r
    return []


def cmd_diag(args) -> int:
    data = _load(args.json)
    yt = _extract_series(data, "y_true")
    yp = _extract_series(data, "y_pred")
    ya = _extract_series(data, "y_all") or None
    if not yt or not yp:
        print("[mbench] 未找到 y_true/y_pred 序列。支持字段：y_true, y_pred, y_all",
              file=sys.stderr)
        return 2
    d = NegativeR2Diagnoser.analyze(yt, yp, ya)
    print(NegativeR2Diagnoser.verdict(d))
    return 0


def cmd_checklist(args) -> int:
    cl = SubmissionChecklist()
    md = cl.to_markdown(results_dir=args.dir, results_json=args.json)
    out = args.output or "mbench_checklist.md"
    Path(out).write_text(md, encoding="utf-8")
    print(f"[mbench] 检查清单已生成: {out}（通过 {md.count('✅')}/{len(cl.items)}）")
    return 0


def cmd_stats(args) -> int:
    a, b = _load(args.a), _load(args.b)
    # 期望每个 JSON 是 {"metric": [seed1, seed2, ...]} 或直接数组
    if "metrics" in a and isinstance(a["metrics"], dict) and isinstance(b.get("metrics"), dict):
        keys = sorted(set(a["metrics"]) & set(b["metrics"]))
        for k in keys:
            va, vb = a["metrics"][k], b["metrics"][k]
            if isinstance(va, list) and isinstance(vb, list):
                _print_compare(k, va, vb)
    elif isinstance(a, list) and isinstance(b, list):
        _print_compare("series", a, b)
    else:
        print("[mbench] stats 需要 {\"metrics\": {key: [seed...]}} 或两个数组", file=sys.stderr)
        return 2
    return 0


def _print_compare(name: str, a: list, b: list) -> None:
    pt = paired_t(a, b)
    bayes = hierarchical_diff(a, b)
    print(f"--- {name} ---")
    print(f"  A: {mean_ci(a)}")
    print(f"  B: {mean_ci(b)}")
    print(f"  paired t: t={pt['t']:.3f} p={pt['p']:.4f} d={pt['d']:.3f} (n={pt['n']})")
    print(f"  bayes P(A>B)={bayes['p_a_gt_b']:.3f} mean_diff={bayes['mean_diff']:.4f} "
          f"HDI=[{bayes['hdi_low']:.4f}, {bayes['hdi_high']:.4f}]")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mbench", description="制造预测 ML 评估协议工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diag", help="负 R² 诊断")
    d.add_argument("json", help="结果 JSON（含 y_true/y_pred）")
    d.set_defaults(func=cmd_diag)

    c = sub.add_parser("checklist", help="投稿检查清单")
    c.add_argument("--dir", default=None, help="结果目录（扫描 *.json）")
    c.add_argument("--json", default=None, help="单结果文件")
    c.add_argument("--output", default=None, help="输出 md 路径")
    c.set_defaults(func=cmd_checklist)

    s = sub.add_parser("stats", help="两组跨 seed 指标对比")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_stats)

    sub.add_parser("version").set_defaults(func=lambda a: print(__version__) or 0)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
