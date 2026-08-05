"""manufacturing-bench: 制造预测 ML 的评估协议产品化。

统一训练协议 + 统计显著性检验 + 负 R² 诊断器 + 投稿检查清单。

教训来源（灵境制造 DL-LNN 论文四阶段实验）：
- smoke 强信号不可信，一切以全量（多 seeds）为准
- 求解器/种子/数据源必须诚信记录
- 近常数目标段 R² 失效，需 var_true_ratio 诊断
- 负 R² 往往不是模型崩溃，而是指标不适用
"""

__version__ = "0.1.0"

from mbench.protocol import ExperimentProtocol  # noqa: F401
from mbench.stats import (  # noqa: F401
    welch_t, paired_t, cohens_d, mean_ci, hierarchical_diff,
)
from mbench.diagnostics import NegativeR2Diagnoser  # noqa: F401
from mbench.checklist import SubmissionChecklist  # noqa: F401

__all__ = [
    "ExperimentProtocol", "NegativeR2Diagnoser", "SubmissionChecklist",
    "welch_t", "paired_t", "cohens_d", "mean_ci", "hierarchical_diff",
]
