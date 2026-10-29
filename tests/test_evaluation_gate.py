import copy

from conftest import ROOT

from fieldassist.gate import GateConfig, compare

GATE = GateConfig.load(ROOT / "config" / "gate.yaml")


def test_gate_config_matches_the_documented_policy():
    assert GATE.max_regression_points == 2.0 and GATE.min_questions == 300


def test_production_configuration_on_the_synthetic_set(reports):
    m = reports["production"]["metrics"]
    assert m["overall"]["questions"] == 300
    assert m["overall"]["accuracy"] >= 99.0 and m["overall"]["recall_at_k"] >= 99.0
    assert m["overall"]["citations_active"] == 100.0
    assert m["slices"]["out_of_scope"]["accuracy"] == 100.0


def test_vector_only_is_blocked_on_fault_codes(reports):
    result = compare(reports["production"], reports["vector-only"], GATE)
    assert result.decision == "block"
    assert any(r.startswith("fault_code accuracy fell") for r in result.reasons)
    m = reports["vector-only"]["metrics"]["slices"]
    assert m["fault_code"]["accuracy"] < 50 and m["spec"]["accuracy"] >= 95


def test_dropping_the_revision_filter_is_blocked_on_superseded_values(reports):
    result = compare(reports["production"], reports["no-revision-filter"], GATE)
    assert result.decision == "block"
    assert any(r.startswith("superseded accuracy fell") for r in result.reasons)
    assert any(r.startswith("citations to current text fell") for r in result.reasons)


def test_same_configuration_with_a_new_model_image_passes(reports):
    candidate = copy.deepcopy(reports["production"])
    candidate["model_image"] = candidate["model_image"].replace("a" * 64, "b" * 64)
    assert compare(reports["production"], candidate, GATE).decision == "pass"


def _with_slice(report, name, accuracy):
    r = copy.deepcopy(report)
    r["metrics"]["slices"][name]["accuracy"] = accuracy
    return r


def test_small_drop_passes_and_drop_beyond_two_points_blocks(reports):
    base = _with_slice(reports["production"], "torque", 95.0)
    assert compare(base, _with_slice(base, "torque", 93.4), GATE).decision == "pass"
    blocked = compare(base, _with_slice(base, "torque", 92.9), GATE)
    assert blocked.decision == "block" and blocked.reasons == ["torque accuracy fell 2.1 points (95.0 -> 92.9)"]


def test_runs_on_different_golden_sets_are_not_comparable(reports):
    other = copy.deepcopy(reports["production"])
    other["golden_hash"] = "0" * 16
    assert "golden sets differ" in compare(reports["production"], other, GATE).reasons[0]


def test_a_partial_run_cannot_pass(reports):
    short = copy.deepcopy(reports["production"])
    short["metrics"]["overall"]["questions"] = 120
    assert compare(reports["production"], short, GATE).decision == "block"
