"""投稿检查清单（SubmissionChecklist）。

四阶段实验教训编码为可核查清单。generate() 扫描结果目录/JSON，
输出 markdown 报告；CLI 一键生成（``mbench checklist``）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# 检查项：(id, 检查内容, 如何核查, 教训来源)
_ITEMS: List[Dict] = [
    {"id": "S1", "check": "seeds ≥ 3（结论以全量为准）",
     "how": "结果 JSON 的 seeds 字段或 runs 数量 ≥ 3",
     "lesson": "单 seed 结论不可信（smoke 假象教训）"},
    {"id": "S2", "check": "求解器诚信记录（ltc_solver）",
     "how": "结果含 ltc_solver=euler/dopri5 字段",
     "lesson": "dopri5 慢 90 倍；不记录则不可复现"},
    {"id": "S3", "check": "统计检验：配对/独立 t + 效应量",
     "how": "比较表含 p 值（<0.05）与 Cohen's d",
     "lesson": "均值差不算数，显著性与效应量才是"},
    {"id": "S4", "check": "负 R² 已诊断（var_true_ratio）",
     "how": "R² 为负时附 var_true_ratio 与 MAE",
     "lesson": "近常数目标段 R² 失效，诊断协议见 diagnostics"},
    {"id": "S5", "check": "数据来源/授权声明",
     "how": "结果 JSON 的 data_note 字段",
     "lesson": "uniwear 等第三方数据必须声明使用授权"},
    {"id": "S6", "check": "可复现性：固定种子 + 两次运行一致",
     "how": "README/复现指南说明固定种子；建议重跑验证",
     "lesson": "固定种子确定性可复现（阶段 2 实证）"},
    {"id": "S7", "check": "smoke 与全量分离（smoke 仅管线验证）",
     "how": "smoke 配置（小样本/少 epoch）与全量配置分离",
     "lesson": "smoke 强信号不可信，结论以全量为准"},
    {"id": "S8", "check": "基线公平：解析/经验基线未作弊",
     "how": "基线未用测试标签（如避免 Tlusty 预测 Tlusty）",
     "lesson": "阶段 4 基线公平化教训"},
    {"id": "S9", "check": "负面结果如实报告",
     "how": "对比表中无优势也保留并讨论",
     "lesson": "负面结果翻转研究路线（用户学术诚信原则）"},
    {"id": "S10", "check": "指标选择与任务匹配",
     "how": "近常数段用 MAE/RMSE，分布预测用校准/覆盖率",
     "lesson": "任务表述决定指标（阶段 4）"},
    {"id": "S11", "check": "AI 预评审失真检测（LLM-reviewer sanity）",
     "how": "每个负面指控须能定位到论文原句/数据；无原文支撑的指控（虚构文献、"
     "声称缺失实际存在的检验）标记失真并丢弃；失真率>30% 的评审整体不采信",
     "lesson": "本地 14B 苛刻审稿人产生幻觉证据（2026-08-06 实证）；苛刻提示词"
     "放大幻觉而非提升质量"},
    {"id": "S12", "check": "图表与附录完整性",
     "how": "初稿附图表清单（编号+文件对应）；参考文献非空；附录含数据/工具链接",
     "lesson": "苛刻评审暴露真实短板：无 References/图表清单"},
]


class SubmissionChecklist:
    """投稿检查清单：扫描结果目录生成 markdown 报告。"""

    def __init__(self, items: Optional[List[Dict]] = None):
        self.items = items or _ITEMS

    def check(self, results_dir: Optional[str] = None,
              results_json: Optional[str] = None) -> List[Dict]:
        """逐项核查。results_dir：扫描 *.json 结果；results_json：单文件。"""
        paths: List[Path] = []
        if results_json:
            paths.append(Path(results_json))
        elif results_dir:
            paths.extend(sorted(Path(results_dir).glob("*.json")))
        elif Path("protocol_results.json").exists():
            paths.append(Path("protocol_results.json"))

        data = {}
        for p in paths:
            try:
                data[p.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data[p.name] = {}

        report = []
        for item in self.items:
            passed = self._verify(item, data)
            report.append({**item, "passed": passed})
        return report

    def _verify(self, item: Dict, data: Dict) -> bool:
        how = item["how"].lower()
        all_text = json.dumps(data, ensure_ascii=False).lower()
        if item["id"] == "S1":
            return any(
                isinstance(d, dict) and isinstance(d.get("seeds"), list)
                and len(d["seeds"]) >= 3
                for d in data.values()
            )
        if item["id"] == "S2":
            return any("ltc_solver" in (d or {}) for d in data.values())
        if item["id"] == "S4":
            return any("var_true_ratio" in (d or {}) for d in data.values())
        if item["id"] == "S5":
            return any("data_note" in (d or {}) for d in data.values())
        if item["id"] == "S6":
            return any("seed" in (d or {}) or "seeds" in (d or {}) for d in data.values())
        if item["id"] == "S11":
            real = [d for d in data.values() if isinstance(d, dict) and d]
            if not real:
                return False  # 无实际结果记录 = 未做失真检测
            flags = [f for d in real for f in (d.get("llm_review_flags") or [])]
            return len(flags) == 0  # 无 AI 评审失真标记 → 通过；有失真标记 → 失败
        if item["id"] == "S12":
            return any(any(k in (d or {}) for k in ("references", "figures", "fig"))
                       for d in data.values())
        # 默认：启发式检查（关键词出现在结果文本中）
        keywords = {
            "S3": ["p", "cohen", "d", "paired", "t-test", "t_test", "p_value"],
            "S7": ["smoke", "smoke_done", "full"],
            "S8": ["baseline", "tlusty", "baseline_fair"],
            "S9": ["negative", "negative_result", "honest", "limit", "诚实", "负面"],
            "S10": ["mae", "coverage", "calibration", "nll", "var_true_ratio"],
        }
        return any(k in all_text for k in keywords.get(item["id"], []))

    def to_markdown(self, results_dir: Optional[str] = None,
                    results_json: Optional[str] = None) -> str:
        """生成 markdown 检查清单报告。"""
        report = self.check(results_dir, results_json)
        lines = [
            "# 投稿检查清单（manufacturing-bench）",
            "",
            f"生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M')}",
            "",
            "| 项 | 检查内容 | 结果 | 教训来源 |",
            "|---|---|---|---|",
        ]
        for r in report:
            mark = "✅" if r["passed"] else "❌"
            lines.append(f"| {r['id']} | {r['check']} | {mark} | {r['lesson']} |")
        passed = sum(1 for r in report if r["passed"])
        lines += ["", f"**通过 {passed}/{len(report)}**",
                  "", "> 未通过项按教训来源修正后重跑 `mbench checklist`。"]
        return "\n".join(lines)
