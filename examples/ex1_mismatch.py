"""示例 1：模态参数失配协议（演示 ExperimentProtocol + 配对检验）。

自包含（无需 torch/灵境制造）：用简单解析函数模拟"物理基线 + 数据分支"。
真实版本见灵境制造 exp46-48。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mbench import ExperimentProtocol, paired_t, cohens_d, welch_t


def physical_baseline(x, mismatch=0.0):
    """解析基线（Tlusty 的替身）：a_lim = 1/(1+0.1n) 随失配漂移。"""
    return 1.0 / (1.0 + 0.1 * x) * (1 - 0.3 * mismatch)


def data_branch(x, rng):
    """数据分支：从失配观测学到的近似。"""
    return physical_baseline(x, 0.0) + rng.normal(0, 0.01, len(x))


def main():
    proto = ExperimentProtocol(name="ex1_mismatch", seeds=[42, 43, 44],
                               output_dir="results/ex1",
                               data_note="synthetic demo (no external data)")
    for seed in proto.seeds:
        proto.seed_all(seed)
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.5, 8.0, 500)
        # 物理基线带 20% 失配（模拟模态参数不确定）
        y_phys = physical_baseline(x, 0.2)
        # 数据分支学习真实（无失配）边界
        y_dl = data_branch(x, rng)
        y_true = physical_baseline(x, 0.0)
        mae_phys = np.mean(np.abs(y_phys - y_true))
        mae_dl = np.mean(np.abs(y_dl - y_true))
        proto.record(mae_phys=float(mae_phys), mae_dl=float(mae_dl))

    path = proto.save()
    print(f"[ex1] 结果已保存: {path}")
    pa = [r["metrics"]["mae_phys"] for r in proto._log]
    pb = [r["metrics"]["mae_dl"] for r in proto._log]
    print(f"physical MAE={np.mean(pa):.4f}±{np.std(pa):.4f}")
    print(f"dl       MAE={np.mean(pb):.4f}±{np.std(pb):.4f}")
    print(f"paired t: {paired_t(pb, pa)}")
    print(f"welch t : {welch_t(pb, pa)}")


if __name__ == "__main__":
    main()
