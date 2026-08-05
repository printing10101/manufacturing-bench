import json
from pathlib import Path

from mbench.checklist import SubmissionChecklist


def _write_protocol_json(tmp: Path, with_seeds=True, with_solver=True) -> Path:
    p = tmp / "protocol_results.json"
    data = {
        "protocol": "test",
        "seeds": [42, 43, 44] if with_seeds else [42],
        "ltc_solver": "euler" if with_solver else None,
        "data_note": "synthetic (Tlusty)",
        "runs": [{"metrics": {"mae": 0.1}}] * 3,
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_checklist_s1_seeds(tmp_path):
    _write_protocol_json(tmp_path, with_seeds=True)
    cl = SubmissionChecklist()
    report = cl.check(results_dir=str(tmp_path))
    assert any(r["id"] == "S1" and r["passed"] for r in report)


def test_checklist_s1_fails_single_seed(tmp_path):
    _write_protocol_json(tmp_path, with_seeds=False)
    cl = SubmissionChecklist()
    report = cl.check(results_dir=str(tmp_path))
    assert any(r["id"] == "S1" and not r["passed"] for r in report)


def test_checklist_s2_solver(tmp_path):
    _write_protocol_json(tmp_path, with_solver=True)
    cl = SubmissionChecklist()
    report = cl.check(results_dir=str(tmp_path))
    assert any(r["id"] == "S2" and r["passed"] for r in report)


def test_checklist_markdown_contains_table(tmp_path):
    _write_protocol_json(tmp_path)
    cl = SubmissionChecklist()
    md = cl.to_markdown(results_dir=str(tmp_path))
    assert "| 项 | 检查内容 | 结果 | 教训来源 |" in md
    assert "✅" in md or "❌" in md


def test_checklist_empty_dir(tmp_path):
    cl = SubmissionChecklist()
    report = cl.check(results_dir=str(tmp_path))
    assert len(report) == len(cl.items)
    assert all(not r["passed"] for r in report)
