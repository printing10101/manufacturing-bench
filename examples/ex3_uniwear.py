"""示例 3：真实磨损负 R² 诊断（演示 NegativeR2Diagnoser）。

自包含：模拟 uniwear 时间外推段（近常数目标）——复现"R² 巨负但 MAE 小"。
真实版本见灵境制造 exp50。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mbench import ExperimentProtocol, NegativeR2Diagnoser


def main():
    proto = ExperimentProtocol(name="ex3_uniwear", seeds=[42, 43, 44],
                               output_dir="results/ex3",
                               data_note="uniwear (nuaa+phm2010) demo, authorized")
    rng = np.random.RandomState(0)
    # 全量磨损：0.03 → 0.31 单调增（有方差）
    t_all = np.linspace(0, 1, 800)
    wear_all = 0.03 + 0.28 * t_all ** 0.8
    # 时间外推测试段：尾段平坦（近常数）
    t_te = np.linspace(0.92, 1.0, 100)
    y_true = 0.03 + 0.28 * t_te ** 0.8
    y_pred = np.full_like(y_true, y_true.mean())   # 预测均值收缩

    d = NegativeR2Diagnoser.analyze(y_true, y_pred, wear_all)
    print("[ex3] 负 R² 诊断（近常数外推段）:")
    print(NegativeR2Diagnoser.verdict(d))

    proto.record(mae=d["mae"], r2=d["r2"], var_true_ratio=d["var_true_ratio"])
    path = proto.save()
    print(f"[ex3] 结果已保存: {path}")


if __name__ == "__main__":
    main()
