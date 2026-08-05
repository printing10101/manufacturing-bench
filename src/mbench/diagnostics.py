"""负 R² 诊断器（NegativeR2Diagnoser）。

阶段 3 的核心发现：真实磨损外推段 R²=−0.5~−20，但 MAE 绝对水平小——
原因是测试段目标方差趋零（近常数段），R² 分母趋零导致指标失效。

诊断协议：
1. var_true_ratio = var(y_true) / var(y_all)  —— 测试段目标方差占比
2. pred_var_ratio = var(y_pred) / var(y_true) —— 预测方差比
3. bias = mean(y_pred - y_true)                —— 系统偏差
4. 判定：var_true_ratio < 0.1 → R² 不适用（近常数目标），以 MAE/RMSE 为主
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class NegativeR2Diagnoser:
    """负 R² 诊断。用 y_true/y_pred（或从结果 JSON 提取）生成诊断报告。"""

    @staticmethod
    def analyze(y_true: Sequence[float], y_pred: Sequence[float],
                y_all: Optional[Sequence[float]] = None) -> Dict[str, float]:
        """诊断分解。y_all：全量目标（训练+测试），用于方差占比。

        若 y_all 缺省，用 y_true 自身（保守：占比=1，R² 失效判定依赖阈值）。
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if len(yt) != len(yp):
            raise ValueError("y_true 与 y_pred 长度不一致")
        ya = np.asarray(y_all, dtype=float) if y_all is not None else yt

        var_true = yt.var()
        var_all = ya.var()
        var_pred = yp.var()
        bias = float(np.mean(yp - yt))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "r2": r2,
            "mae": float(np.mean(np.abs(yp - yt))),
            "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "var_true_ratio": float(var_true / var_all) if var_all > 0 else 1.0,
            "pred_var_ratio": float(var_pred / var_true) if var_true > 0 else 0.0,
            "bias": bias,
            "n": int(len(yt)),
        }

    @staticmethod
    def verdict(d: Dict[str, float]) -> str:
        """诊断结论：R² 是否适用、模型是否失效。"""
        lines = []
        if d["var_true_ratio"] < 0.1:
            lines.append(
                f"[R² 不适用] 测试段目标方差占比={d['var_true_ratio']:.4f} < 0.1："
                "近常数目标段，R² 分母趋零，任何偏差都导致巨负 R²。"
                "应以 MAE/RMSE 为主指标。"
            )
        else:
            lines.append(
                f"[R² 可用] 目标方差占比={d['var_true_ratio']:.4f} ≥ 0.1，R² 有意义。"
            )
        if abs(d["pred_var_ratio"] - 1.0) > 0.5:
            lines.append(
                f"[方差失配] 预测方差/真实方差={d['pred_var_ratio']:.2f}（偏离 1 超过 0.5）："
                "预测过度收缩或发散。"
            )
        if abs(d["bias"]) > 0.05:
            lines.append(f"[系统偏差] bias={d['bias']:.4f}（超过 0.05，绝对值尺度）")
        lines.append(f"MAE={d['mae']:.4f}, RMSE={d['rmse']:.4f}, R²={d['r2']:.3f}")
        return "\n".join(lines)
