# manufacturing-bench

**制造预测 ML 的评估协议产品化**：统一训练协议 + 统计显著性检验 + 负 R² 诊断器 + 投稿检查清单。

面向制造领域的 ML 研究（颤振预测、刀具磨损、稳定性边界等），把论文实验里
最稀缺、最可复用的**评估协议**打包成零依赖工具（仅 numpy/scipy，无 torch 依赖）。

## 为什么需要它

制造预测论文的常见硬伤（本包每一条教训都来自真实实验踩坑）：

| 硬伤 | 后果 | 本包对策 |
|---|---|---|
| 单 seed 跑一次就下结论 | smoke 强信号在全量下消失（真实教训：−68% 假象 → 全量 −1.6%） | `ExperimentProtocol` 强制 seeds ≥ 3 |
| 求解器不记录 | dopri5 慢 90 倍、结果不可复现 | `ltc_solver` 诚信记录 + `force_euler` |
| 均值差当结论 | 无显著性、无效应量 | `welch_t` / `paired_t` / `cohens_d` / 层级贝叶斯 |
| 近常数段 R² 巨负却说模型崩了 | 指标失效被误读为模型失败 | `NegativeR2Diagnoser` 方差占比分解 |
| 基线用测试标签生成（"Tlusty 预测 Tlusty"） | 作弊基线，审稿人一票否决 | 检查清单 S8 |
| 负面结果不敢写 | 丧失最重要的诚实贡献 | 检查清单 S9（负面结果翻转路线） |

## 安装

```bash
pip install -e .            # 开发安装（含 CLI: mbench）
# 或零安装使用：
export PYTHONPATH=src
py -3.11 -m mbench.cli version
```

## CLI 快速开始

```bash
# 1. 负 R² 诊断（结果 JSON 含 y_true/y_pred/y_all）
mbench diag results.json
# → [R² 不适用] 测试段目标方差占比=0.0002 < 0.1：近常数目标段，R² 分母趋零……
#   MAE=0.0050, RMSE=0.0051, R²=-33.248

# 2. 投稿检查清单（扫描结果目录，一键生成 markdown）
mbench checklist --dir experiments/results --output checklist.md

# 3. 两组跨 seed 指标对比（配对 t + 效应量 + 层级贝叶斯）
mbench stats results_a.json results_b.json
# → paired t: t=-8.660 p=0.0131 d=-7.071
#   bayes P(A>B)=0.004 mean_diff=-0.1000 HDI=[-0.1376, -0.0614]
```

## Python API

```python
from mbench import (ExperimentProtocol, NegativeR2Diagnoser,
                    paired_t, cohens_d, hierarchical_diff)

# 统一训练协议（seeds + 求解器诚信 + 结果落盘）
proto = ExperimentProtocol(name="exp51", seeds=[42, 43, 44],
                           ltc_solver="euler", data_note="synthetic Tlusty")
proto.force_euler(models)          # import models 后调用
for seed in proto.seeds:
    proto.seed_all(seed)
    mae, r2 = train_and_eval()     # 你的训练逻辑
    proto.record(mae=mae, r2=r2)
proto.save()

# 统计检验
print(paired_t(a_maes, b_maes))    # {'t':..., 'p':..., 'd':..., 'n':...}
print(hierarchical_diff(a_maes, b_maes))  # P(A>B) + HDI

# 负 R² 诊断（近常数目标段）
d = NegativeR2Diagnoser.analyze(y_true, y_pred, y_all)
print(NegativeR2Diagnoser.verdict(d))
```

## 示例（examples/）

| 示例 | 场景 | 演示 |
|---|---|---|
| `ex1_mismatch.py` | 模态参数失配（合成 Tlusty） | 协议 + 配对检验 |
| `ex2_extrap.py` | 跨转速外推 | 门控激活 vs 分布外 |
| `ex3_uniwear.py` | 真实刀具磨损（uniwear） | 负 R² 诊断 |

## 复现指南

1. 每个实验脚本声明：种子列表、求解器、数据来源/授权（`data_note`）
2. 结论一律以全量（多 seeds）为准；smoke 仅验证管线
3. 结果 JSON 含全部诚信元数据（`protocol_results.json` 结构）
4. 投稿前跑 `mbench checklist`，10 项全过才提交

## 检查清单（10 项）

S1 seeds≥3 ｜ S2 求解器记录 ｜ S3 显著性+效应量 ｜ S4 负 R² 诊断 ｜
S5 数据授权 ｜ S6 可复现性 ｜ S7 smoke/全量分离 ｜ S8 基线公平 ｜
S9 负面结果如实报告 ｜ S10 指标与任务匹配

## 教训出处

本包教训来自 DL-LNN 铣削颤振论文的四阶段实验（2026-08，详见
`docs/lessons.md`）：参数失配 / 跨转速外推 / uniwear 真实磨损 / SLD 分布预测。

## License

MIT
