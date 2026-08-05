import numpy as np
import pytest

from mbench.diagnostics import NegativeR2Diagnoser


def test_analyze_near_constant_target_flags_r2_unusable():
    """近常数目标段：var_true_ratio 极小 → R² 判定不适用。"""
    y_all = np.random.RandomState(0).uniform(0.1, 0.3, 1000)  # 全量有方差
    y_true = np.full(100, 0.17) + np.random.RandomState(1).normal(0, 0.001, 100)
    y_pred = np.full(100, 0.175)
    d = NegativeR2Diagnoser.analyze(y_true, y_pred, y_all)
    assert d["var_true_ratio"] < 0.01
    assert "不适用" in NegativeR2Diagnoser.verdict(d)
    # R² 可能为负但 MAE 小——诊断要抓住这个本质
    assert d["mae"] < 0.01


def test_analyze_normal_target_r2_usable():
    y_true = np.random.RandomState(0).uniform(0.1, 0.9, 200)
    y_pred = y_true + np.random.RandomState(1).normal(0, 0.05, 200)
    d = NegativeR2Diagnoser.analyze(y_true, y_pred, y_true)
    assert d["var_true_ratio"] >= 1.0
    assert "可用" in NegativeR2Diagnoser.verdict(d)


def test_analyze_identical_length_required():
    with pytest.raises(ValueError):
        NegativeR2Diagnoser.analyze([1, 2, 3], [1, 2])


def test_analyze_perfect_prediction():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    d = NegativeR2Diagnoser.analyze(y, y, y)
    assert d["r2"] == 1.0
    assert d["mae"] == 0.0
