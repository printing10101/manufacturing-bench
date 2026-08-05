"""统一训练协议（ExperimentProtocol）。

封装四阶段实验沉淀的关键约定：
1. seeds 管理：torch/np/cuda 统一播种，结果按 seed 记录
2. 求解器诚信记录：ODE 求解器（euler/dopri5）强制声明，写入结果
3. smoke/full 分级：小规模 smoke 仅用于管线验证，结论一律以全量为准
4. 可复现性：同一配置两次运行应逐位一致（固定种子 + 确定性算子）
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class ExperimentProtocol:
    """实验协议：播种 + 求解器记录 + 分级验证 + 结果落盘。"""

    name: str
    seeds: list = field(default_factory=lambda: [42, 43, 44])
    ltc_solver: str = "euler"          # 诚信记录：ODE 求解器
    output_dir: str = "results"
    data_note: str = ""                # 数据来源/授权记录

    def __post_init__(self) -> None:
        self._log: list[Dict[str, Any]] = []
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 播种 ----------
    def seed_all(self, seed: int) -> None:
        """torch/np/random 统一播种（torch 可选）。"""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # ---------- 求解器 ----------
    @staticmethod
    def force_euler(models_module: Any) -> str:
        """强制 Euler 求解器（dopri5 实测慢 90 倍）。

        用法：``ExperimentProtocol.force_euler(models)`` 在 import models 后调用。
        返回实际使用的求解器名。
        """
        flag = getattr(models_module, "_HAS_TORCHDIFFEQ", None)
        if flag is not None:
            models_module._HAS_TORCHDIFFEQ = False
        return "euler"

    # ---------- 分级验证 ----------
    @staticmethod
    def run_smoke(run_full: Callable[[], None], **overrides: Any) -> None:
        """smoke 运行：小规模管线验证（结论一律以全量为准，见 README 教训）。"""
        for k, v in overrides.items():
            setattr(run_full.__globals__.get("__package__") or __import__("__main__"), k, v)
        t0 = time.time()
        run_full()
        print(f"[mbench] smoke done in {time.time() - t0:.1f}s "
              f"(overrides={overrides})", flush=True)

    # ---------- 结果记录 ----------
    def record(self, **metrics: Any) -> None:
        """记录一次（seed）的指标，自动附诚信元数据。"""
        entry = {
            "protocol": self.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ltc_solver": self.ltc_solver,
            "data_note": self.data_note,
            "metrics": metrics,
        }
        self._log.append(entry)

    def save(self, filename: str = "protocol_results.json") -> Path:
        """结果落盘（含种子/求解器/数据诚信记录）。"""
        out = {
            "protocol": self.name,
            "seeds": self.seeds,
            "ltc_solver": self.ltc_solver,
            "data_note": self.data_note,
            "runs": self._log,
        }
        path = self.output_dir / filename
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def summary(self, key: str) -> Dict[str, Any]:
        """对某指标输出跨 seed 均值/标准差。"""
        vals = [r["metrics"][key] for r in self._log if key in r["metrics"]]
        if not vals:
            return {}
        return {
            "key": key,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
