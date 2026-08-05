import numpy as np
import pytest

from mbench.stats import cohens_d, hierarchical_diff, mean_ci, paired_t, welch_t


def test_welch_t_significant():
    a = np.random.RandomState(0).normal(0.5, 0.1, 30)
    b = np.random.RandomState(1).normal(0.2, 0.1, 30)
    r = welch_t(a, b)
    assert r["p"] < 0.05
    assert r["d"] > 1.0


def test_welch_t_nonsignificant():
    a = np.random.RandomState(0).normal(0.5, 0.5, 10)
    b = np.random.RandomState(1).normal(0.5, 0.5, 10)
    r = welch_t(a, b)
    assert r["p"] > 0.05


def test_paired_t_equal_length():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 3.5, 4.5]
    r = paired_t(a, b)
    assert r["n"] == 4
    assert r["p"] < 0.05


def test_paired_t_requires_equal_length():
    with pytest.raises(ValueError):
        paired_t([1, 2], [1, 2, 3])


def test_cohens_d_identical_zero():
    a = [1.0, 2.0, 3.0]
    assert cohens_d(a, a) == 0.0


def test_cohens_d_known_direction():
    a = [1.0, 1.1, 0.9, 1.05]
    b = [5.0, 5.1, 4.9, 5.05]
    assert cohens_d(a, b) < 0  # b 更大 → d 为负


def test_mean_ci_contains_sample_mean():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = mean_ci(a)
    assert r["mean"] == 3.0
    assert r["ci_low"] < r["mean"] < r["ci_high"]


def test_hierarchical_diff_separated_groups():
    a = np.random.RandomState(0).normal(1.0, 0.2, 12)
    b = np.random.RandomState(1).normal(0.0, 0.2, 12)
    r = hierarchical_diff(a, b, n_samples=4000)
    assert r["p_a_gt_b"] > 0.9
    assert r["mean_diff"] > 0.5


def test_hierarchical_diff_overlapping_groups():
    a = np.random.RandomState(0).normal(0.5, 1.0, 12)
    b = np.random.RandomState(1).normal(0.5, 1.0, 12)
    r = hierarchical_diff(a, b, n_samples=4000)
    assert 0.01 < r["p_a_gt_b"] < 0.99
    assert abs(r["mean_diff"]) < 1.0
