"""示例 2：跨转速外推协议（演示门控/分布外对比的统计检验）。

自包含：模拟"训练转速内 vs 外推区"的数据分支误差差异。
真实版本见灵境制造 exp49。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mbench import ExperimentProtocol, paired_t, hierarchical_diff


def train_in_range(rng, n):
    return rng.normal(0, 0.05, n)          # 域内：数据分支小误差


def extrap_range(rng, n):
    return rng.normal(0, 0.45, n)          # 外推：数据分支大漂移（LSTM 崩溃）


def main():
    proto = ExperimentProtocol(name="ex2_extrap", seeds=[42, 43, 44],
                               output_dir="results/ex2",
                               data_note="synthetic demo (spindle extrapolation)")
    for seed in proto.seeds:
        proto.seed_all(seed)
        rng = np.random.RandomState(seed)
        err_in = train_in_range(rng, 100)
        err_ex = extrap_range(rng, 100)
        proto.record(err_in=float(np.mean(err_in ** 2)),
                     err_ex=float(np.mean(err_ex ** 2)))

    path = proto.save()
    print(f"[ex2] 结果已保存: {path}")
    ain = [r["metrics"]["err_in"] for r in proto._log]
    aex = [r["metrics"]["err_ex"] for r in proto._log]
    print(f"in-domain MSE={np.mean(ain):.4f}±{np.std(ain):.4f}")
    print(f"extrap    MSE={np.mean(aex):.4f}±{np.std(aex):.4f}")
    print(f"paired t (in vs extrap): {paired_t(ain, aex)}")
    print(f"bayes P(extrap > in): {hierarchical_diff(aex, ain)}")


if __name__ == "__main__":
    main()
