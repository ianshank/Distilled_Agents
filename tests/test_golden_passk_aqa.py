"""Unit and regression tests for Pass@K hard/OOD gate, Chen estimator, and golden set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.golden import (
    BUCKET_MINIMA,
    TOTAL_MINIMUM,
    validate_golden_file,
    validate_golden_rows,
)
from enhanced_system.harness.score import (
    calculate_pass_at_k,
    estimate_pass_at_k,
)
from enhanced_system.harness.types import (
    GoldenRow,
    HarnessRunResult,
    Trajectory,
)
from scripts.harness import eval_harness as eval_harness_module
from scripts.harness.check_rule_matrix import verify_rule_matrix
from scripts.harness.run_pass_at_k import evaluate_pass_at_k
from scripts.harness.run_pass_at_k import main as run_pass_k_main


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_schema_backwards_compatibility() -> None:
    """Existing golden rows with only prompt and expected load with correct defaults."""
    raw = {"prompt": "What is Python?", "expected": "A programming language"}
    row = GoldenRow.model_validate(raw)
    assert row.prompt == "What is Python?"
    assert row.expected == "A programming language"
    assert row.slice == "core"
    assert row.grader == "exact"
    assert row.allow_semantic is False
    assert row.id is None
    assert row.harness_id is None
    assert row.expected_tools is None
    assert row.is_ood is False


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_empty_prompt_and_slice_normalize() -> None:
    """Empty prompts are rejected and slice casing is normalized."""
    with pytest.raises(ValueError, match="must not be empty"):
        GoldenRow.model_validate({"prompt": ""})

    with pytest.raises(ValueError, match="must not be empty"):
        GoldenRow.model_validate({"prompt": "   "})

    row_upper = GoldenRow.model_validate({"prompt": "Valid", "slice": "HARD"})
    assert row_upper.slice == "hard"

    row_core_upper = GoldenRow.model_validate({"prompt": "Valid", "slice": "CORE"})
    assert row_core_upper.slice == "core"

    row_ood_upper = GoldenRow.model_validate({"prompt": "Valid", "slice": "OOD"})
    assert row_ood_upper.slice == "ood"

    row_none = GoldenRow.model_validate({"prompt": "Valid", "slice": None})
    assert row_none.slice == "core"


@pytest.mark.unit
@pytest.mark.harness
def test_ood_detector_equivalence() -> None:
    """A row is OOD iff slice == 'ood' OR ood == true (equivalent handling)."""
    # slice: "ood", ood: False -> OOD
    row1 = GoldenRow.model_validate({"prompt": "Task", "slice": "ood", "ood": False})
    assert row1.is_ood is True

    # slice: "hard", ood: True -> OOD
    row2 = GoldenRow.model_validate({"prompt": "Task", "slice": "hard", "ood": True})
    assert row2.is_ood is True

    # slice: "ood", ood: True -> OOD
    row3 = GoldenRow.model_validate({"prompt": "Task", "slice": "ood", "ood": True})
    assert row3.is_ood is True

    # slice: "hard", ood: False -> NOT OOD
    row4 = GoldenRow.model_validate({"prompt": "Task", "slice": "hard", "ood": False})
    assert row4.is_ood is False

    # slice: "core", ood: False -> NOT OOD
    row5 = GoldenRow.model_validate({"prompt": "Task", "slice": "core", "ood": False})
    assert row5.is_ood is False


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_hard_and_ood_validation() -> None:
    """Hard and OOD slice rows enforce exact match, non-empty expected, and canonical refusal for OOD."""
    valid_hard = {
        "id": "sqe-hard-001",
        "prompt": "DAG topo order",
        "expected": "checkout compile",
        "slice": "hard",
        "grader": "exact",
        "allow_semantic": False,
    }
    row_hard = GoldenRow.model_validate(valid_hard)
    row_hard.validate_for_hard_or_ood()

    valid_ood = {
        "id": "sqe-ood-001",
        "prompt": "Cycle",
        "expected": "BLOCKED:CYCLE_DETECTED",
        "slice": "ood",
        "grader": "exact",
        "allow_semantic": False,
    }
    row_ood = GoldenRow.model_validate(valid_ood)
    row_ood.validate_for_hard_or_ood()

    # OOD with non-canonical expected raises ValueError
    invalid_ood = {
        "id": "sqe-ood-bad",
        "prompt": "Cycle",
        "expected": "Some fabricated order",
        "slice": "ood",
        "grader": "exact",
        "allow_semantic": False,
    }
    row_bad_ood = GoldenRow.model_validate(invalid_ood)
    with pytest.raises(ValueError, match="canonical refusal"):
        row_bad_ood.validate_for_hard_or_ood()

    # Hard row with allow_semantic=True raises ValueError
    semantic_hard = {
        "id": "sqe-hard-sem",
        "prompt": "DAG",
        "expected": "out",
        "slice": "hard",
        "allow_semantic": True,
    }
    row_sem = GoldenRow.model_validate(semantic_hard)
    with pytest.raises(ValueError, match="allow_semantic=False"):
        row_sem.validate_for_hard_or_ood()

    row_core = GoldenRow.model_validate({"id": "core-001", "prompt": "Core", "expected": "ok"})
    with pytest.raises(ValueError, match="must have slice in"):
        row_core.validate_for_hard_or_ood()


@pytest.mark.unit
@pytest.mark.harness
def test_chen_unbiased_pass_at_k_math() -> None:
    """Verify Chen et al. unbiased estimator against analytical values and biased formula."""
    # c = 0 -> 0.0
    assert estimate_pass_at_k(n=5, c=0, k=3) == 0.0

    # c = n -> 1.0
    assert estimate_pass_at_k(n=5, c=5, k=3) == 1.0

    # c = 3, n = 5, k = 3: n - c = 2 < 3 -> 1.0
    assert estimate_pass_at_k(n=5, c=3, k=3) == 1.0

    # c = 2, n = 5, k = 3: 1 - comb(3, 3) / comb(5, 3) = 1 - 1/10 = 0.9
    assert pytest.approx(estimate_pass_at_k(n=5, c=2, k=3)) == 0.9

    # c = 1, n = 5, k = 3: 1 - comb(4, 3) / comb(5, 3) = 1 - 4/10 = 0.6
    assert pytest.approx(estimate_pass_at_k(n=5, c=1, k=3)) == 0.6

    # k = 1 is always exactly c / n
    for c in range(6):
        assert pytest.approx(estimate_pass_at_k(n=5, c=c, k=1)) == c / 5.0

    # Prove discrepancy with biased estimator 1 - (1 - c/n)^k
    # When c=1, n=5, k=3: unbiased is 0.6; biased is 1 - (4/5)^3 = 1 - 64/125 = 0.488
    biased_val = 1.0 - (1.0 - 1.0 / 5.0) ** 3
    unbiased_val = estimate_pass_at_k(n=5, c=1, k=3)
    assert unbiased_val != biased_val
    assert pytest.approx(unbiased_val) == 0.6
    assert pytest.approx(biased_val) == 0.488

    # Parameter validation
    with pytest.raises(ValueError, match="n=2 must be >= k=3"):
        estimate_pass_at_k(n=2, c=1, k=3)

    with pytest.raises(ValueError, match="k must be positive"):
        estimate_pass_at_k(n=5, c=1, k=0)

    with pytest.raises(ValueError, match="Correct samples"):
        estimate_pass_at_k(n=5, c=6, k=3)


@pytest.mark.unit
@pytest.mark.harness
def test_calculate_pass_at_k_multi_problem() -> None:
    """Test average pass@k across a batch of problems."""
    # Problem 1: 5 samples, 1 correct (pass@3 = 0.6)
    # Problem 2: 5 samples, 2 correct (pass@3 = 0.9)
    # Average pass@3 = (0.6 + 0.9) / 2 = 0.75
    counts = [(5, 1), (5, 2)]
    avg_pass_3 = calculate_pass_at_k(counts, k=3)
    assert pytest.approx(avg_pass_3) == 0.75

    avg_pass_1 = calculate_pass_at_k(counts, k=1)
    assert pytest.approx(avg_pass_1) == (0.2 + 0.4) / 2.0


@pytest.mark.unit
@pytest.mark.harness
def test_golden_file_bucket_minima_passes() -> None:
    """Checked-in configs/golden_sets/sqe_hard_ood.jsonl meets all bucket minima."""
    result = validate_golden_file("configs/golden_sets/sqe_hard_ood.jsonl")
    assert result["valid"] is True
    assert result["total"] >= TOTAL_MINIMUM
    counts = result["bucket_counts"]
    for bucket, minimum in BUCKET_MINIMA.items():
        assert counts.get(bucket, 0) >= minimum, (
            f"Bucket {bucket} has {counts.get(bucket, 0)} rows, expected >= {minimum}"
        )


@pytest.mark.unit
@pytest.mark.harness
def test_golden_validator_catches_violations() -> None:
    """Validator raises ValueError on deficient count, missing bucket, or duplicates."""
    # Less than total minimum (24)
    short_rows = [
        GoldenRow(
            id=f"row-{i}",
            prompt=f"Task {i}",
            expected="ok",
            slice="hard",
            bucket="dag",
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="minimum required is 24"):
        validate_golden_rows(short_rows)

    # 24 rows but missing cycle/unsat buckets (all DAG)
    all_dag_rows = [
        GoldenRow(
            id=f"dag-{i}",
            prompt=f"DAG Task {i}",
            expected="ok",
            slice="hard",
            bucket="dag",
        )
        for i in range(25)
    ]
    with pytest.raises(ValueError, match="failed bucket minima validation"):
        validate_golden_rows(all_dag_rows)


@pytest.mark.unit
@pytest.mark.harness
def test_echo_backend_reset_and_multi_trial() -> None:
    """EchoBackend correctly handles dict mapping, sequential trial queues, and reset."""
    mapping = {
        "task_one": [
            '{"tool": "final_answer", "args": {"text": "trial1"}}',
            '{"tool": "final_answer", "args": {"text": "trial2"}}',
        ],
        "default": '{"tool": "final_answer", "args": {"text": "fallback"}}',
    }
    backend = EchoBackend(mapping)

    res1 = backend.generate([{"role": "user", "content": "Execute task_one"}])
    assert "trial1" in res1[0]
    res2 = backend.generate([{"role": "user", "content": "Execute task_one"}])
    assert "trial2" in res2[0]

    # After reset, queue is reloaded
    backend.reset()
    res1_reset = backend.generate([{"role": "user", "content": "Execute task_one"}])
    assert "trial1" in res1_reset[0]


@pytest.mark.unit
@pytest.mark.harness
def test_ood_synthetic_success_prohibition(tmp_path: Path) -> None:
    """Any non-empty non-matching answer on an OOD row trips synthetic-success violation."""
    # OOD row expecting BLOCKED:CYCLE_DETECTED
    row = GoldenRow(
        id="sqe-ood-cycle-test",
        prompt="Circular dependency",
        expected="BLOCKED:CYCLE_DETECTED",
        slice="ood",
        ood=True,
    )

    # Case 1: Backend emits invented topological order "a b c" (non-empty non-matching)
    invented_backend = EchoBackend(
        {"Circular dependency": '{"tool": "final_answer", "args": {"text": "a b c"}}'}
    )
    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    runtime_invented = AgentRuntime(invented_backend, spec=load_spec("base_react"), teacher=False)
    res_invented = evaluate_pass_at_k(
        runtime_invented,
        [row],
        n=3,
        k=2,
        harness_id="base_react",
    )
    # Must trip 3 violations (one per sample)
    assert res_invented["ood_synthetic_success_violations"] == 3
    assert res_invented["pass_at_k"]["2"] == 0.0

    # Case 2: Backend emits canonical refusal BLOCKED:CYCLE_DETECTED
    canonical_backend = EchoBackend(
        {
            "Circular dependency": (
                '{"tool": "final_answer", "args": {"text": "BLOCKED:CYCLE_DETECTED"}}'
            )
        }
    )
    runtime_canonical = AgentRuntime(canonical_backend, spec=load_spec("base_react"), teacher=False)
    res_canonical = evaluate_pass_at_k(
        runtime_canonical,
        [row],
        n=3,
        k=2,
        harness_id="base_react",
    )
    assert res_canonical["ood_synthetic_success_violations"] == 0
    assert res_canonical["pass_at_k"]["2"] == 1.0


@pytest.mark.unit
@pytest.mark.harness
def test_ood_security_failure_is_not_synthetic_success() -> None:
    """Canonical OOD refusals with security findings stay non-violations."""

    class StubRuntime:
        def run(
            self, task: str, *, harness_id: str | None = None, temperature: float | None = None
        ):
            return HarnessRunResult(
                final_answer="BLOCKED:CYCLE_DETECTED",
                harness_id=harness_id or "",
                trajectory=Trajectory(harness_id=harness_id or ""),
            )

    class StubScanner:
        def scan_trajectory(self, trajectory: Trajectory) -> list[dict[str, str]]:
            return [{"issue_text": "simulated"}]

    row = GoldenRow(
        id="sqe-ood-cycle-test",
        prompt="Circular dependency",
        expected="BLOCKED:CYCLE_DETECTED",
        slice="ood",
        ood=True,
    )
    result = evaluate_pass_at_k(
        StubRuntime(),
        [row],
        n=2,
        k=1,
        harness_id="base_react",
        scanner=StubScanner(),
    )
    assert result["ood_synthetic_success_violations"] == 0
    assert result["pass_at_k"]["1"] == 0.0


@pytest.mark.unit
@pytest.mark.harness
def test_pass_at_k_uses_row_harness_and_temperature() -> None:
    """Pass@K runs honor per-row harness_id and forwarded sampling temperature."""

    class StubRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None, float | None]] = []

        def run(
            self, task: str, *, harness_id: str | None = None, temperature: float | None = None
        ):
            self.calls.append((task, harness_id, temperature))
            return HarnessRunResult(
                final_answer="ok",
                harness_id=harness_id or "",
                trajectory=Trajectory(harness_id=harness_id or ""),
            )

    runtime = StubRuntime()
    rows = [
        GoldenRow(
            id="hard-1", prompt="task-1", expected="ok", slice="hard", harness_id="row_harness"
        ),
        GoldenRow(id="hard-2", prompt="task-2", expected="ok", slice="hard"),
    ]
    evaluate_pass_at_k(runtime, rows, n=1, k=1, harness_id="cli_harness", temperature=0.8)
    assert runtime.calls == [
        ("task-1", "row_harness", 0.8),
        ("task-2", "cli_harness", 0.8),
    ]


@pytest.mark.unit
@pytest.mark.harness
def test_vacuity_guard_gate_can_fail(tmp_path: Path) -> None:
    """Verify that run_pass_at_k.py exits non-zero when threshold fails or OOD is violated."""
    golden_file = tmp_path / "test_golden.jsonl"
    golden_file.write_text(
        json.dumps(
            {
                "id": "sqe-ood-test",
                "prompt": "Circular dep",
                "expected": "BLOCKED:CYCLE_DETECTED",
                "slice": "ood",
                "bucket": "cycle",
                "grader": "exact",
                "allow_semantic": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # 1. Failing run: threshold 1.0, bad responses
    scripted_file = tmp_path / "failing_scripted.json"
    scripted_file.write_text(
        json.dumps(
            {"Circular dep": '{"tool": "final_answer", "args": {"text": "invented order"}}'}
        ),
        encoding="utf-8",
    )

    # Should exit 1 because golden set has < 24 rows
    exit_code_small = run_pass_k_main(
        [
            "--golden-set",
            str(golden_file),
            "--scripted",
            str(scripted_file),
        ]
    )
    assert exit_code_small == 1

    assert run_pass_k_main(["--threshold", "-1"]) == 1
    assert run_pass_k_main(["--threshold", "101"]) == 1


@pytest.mark.unit
@pytest.mark.harness
def test_rule_matrix_traceability() -> None:
    """configs/rule_traceability/matrix.yaml covers all hard golden IDs in sqe_hard_ood.jsonl."""
    matrix_path = Path("configs/rule_traceability/matrix.yaml")
    golden_path = Path("configs/golden_sets/sqe_hard_ood.jsonl")
    result = verify_rule_matrix(matrix_path, golden_path)
    assert result["covered_hard_ids"] == 16
    assert result["total_rules"] >= 16


@pytest.mark.unit
@pytest.mark.harness
def test_rule_matrix_rejects_missing_hard_id(tmp_path: Path) -> None:
    """Hard rows without ids fail traceability verification."""
    matrix_path = tmp_path / "matrix.yaml"
    golden_path = tmp_path / "golden.jsonl"
    matrix_path.write_text(
        "rules:\n  - rule_id: r1\n    source_trace: trace\n    harness_id: base_react\n    golden_id: hard-1\n    solver_status: fixture\n",
        encoding="utf-8",
    )
    golden_path.write_text(
        json.dumps({"prompt": "task", "expected": "ok", "slice": "hard"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing a stable 'id'"):
        verify_rule_matrix(matrix_path, golden_path)


@pytest.mark.unit
@pytest.mark.harness
def test_eval_harness_uses_row_harness_and_exact_rows_stay_strict(tmp_path: Path) -> None:
    """Single-pass evaluation honors row harness ids and blocks semantic-only exact rows."""

    class StubRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def run(self, task: str, *, harness_id: str | None = None):
            self.calls.append((task, harness_id))
            answer = "context containers components and code"
            return HarnessRunResult(
                final_answer=answer,
                harness_id=harness_id or "",
                trajectory=Trajectory(harness_id=harness_id or ""),
            )

    input_path = tmp_path / "eval.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "semantic exact row",
                        "expected": "Context, Containers, Components, and Code",
                        "slice": "core",
                        "allow_semantic": True,
                        "grader": "exact",
                    }
                ),
                json.dumps(
                    {
                        "prompt": "semantic non-exact row",
                        "expected": "Context, Containers, Components, and Code",
                        "slice": "core",
                        "allow_semantic": True,
                        "grader": "semantic",
                        "harness_id": "row_harness",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = StubRuntime()
    result = eval_harness_module._evaluate_file(
        runtime,
        input_path,
        "cli_harness",
        strict=True,
        scanner=None,
    )
    assert runtime.calls == [
        ("semantic exact row", "cli_harness"),
        ("semantic non-exact row", "row_harness"),
    ]
    assert result["exact_match"] == 0
    assert result["semantic_match"] == 1
    assert result["pass_rate"] == 50.0


@pytest.mark.unit
@pytest.mark.harness
def test_golden_infer_bucket_and_file_errors(tmp_path: Path) -> None:
    """infer_bucket_from_row edge cases and file error handling."""
    from enhanced_system.harness.golden import infer_bucket_from_row

    # Inferred from tags
    assert infer_bucket_from_row({"prompt": "t", "tags": ["dag"]}) == "dag"
    # Inferred from ID
    assert infer_bucket_from_row({"prompt": "t", "id": "test_tree_01"}) == "condition_tree"
    assert infer_bucket_from_row({"prompt": "t", "id": "test_schema_01"}) == "schema_syntax"
    assert infer_bucket_from_row({"prompt": "t", "id": "test_theory_01"}) == "unknown_theory"
    assert infer_bucket_from_row({"prompt": "t", "id": "unknown_id"}) == "unknown"

    # Duplicate row ID error
    dups = [
        GoldenRow(id="dup", prompt="p1", expected="e", slice="hard", bucket="dag"),
        GoldenRow(id="dup", prompt="p2", expected="e", slice="hard", bucket="dag"),
    ] + [
        GoldenRow(id=f"id-{i}", prompt=f"p{i}", expected="e", slice="hard", bucket="dag")
        for i in range(25)
    ]
    with pytest.raises(ValueError, match="Duplicate row id detected"):
        validate_golden_rows(dups)

    # File not found
    with pytest.raises(FileNotFoundError):
        validate_golden_file(tmp_path / "non_existent.jsonl")

    # Invalid JSON in file
    bad_json_file = tmp_path / "bad.jsonl"
    bad_json_file.write_text("invalid json line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        validate_golden_file(bad_json_file)


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_invariants_failure_paths() -> None:
    """Test all error branches of validate_for_hard_slice and validate_for_hard_or_ood."""
    # validate_for_hard_slice
    with pytest.raises(ValueError, match="must have slice='hard'"):
        GoldenRow(prompt="p", expected="e", slice="core").validate_for_hard_slice()

    with pytest.raises(ValueError, match="missing stable id"):
        GoldenRow(prompt="p", expected="e", slice="hard").validate_for_hard_slice()

    with pytest.raises(ValueError, match="must have non-empty expected"):
        GoldenRow(id="h1", prompt="p", expected=None, slice="hard").validate_for_hard_slice()

    # validate_for_hard_or_ood
    with pytest.raises(ValueError, match="must have slice in"):
        GoldenRow(id="h1", prompt="p", expected="e", slice="core").validate_for_hard_or_ood()

    with pytest.raises(ValueError, match="missing stable id"):
        GoldenRow(prompt="p", expected="e", slice="hard").validate_for_hard_or_ood()

    with pytest.raises(ValueError, match="must have non-empty expected"):
        GoldenRow(id="h1", prompt="p", expected=None, slice="hard").validate_for_hard_or_ood()

    with pytest.raises(ValueError, match="must have grader='exact'"):
        GoldenRow(
            id="h1", prompt="p", expected="e", slice="hard", grader="semantic"
        ).validate_for_hard_or_ood()


@pytest.mark.unit
@pytest.mark.harness
def test_echo_backend_dict_object_and_empty() -> None:
    """EchoBackend handles dictionary return values and default fallbacks."""
    backend = EchoBackend(
        {
            "dict_task": {"tool": "final_answer", "args": {"text": "dict_res"}},
            "default": {"tool": "final_answer", "args": {"text": "default_res"}},
        }
    )
    res = backend.generate([{"role": "user", "content": "Execute dict_task"}])
    assert "dict_res" in res[0]

    res_def = backend.generate([{"role": "user", "content": "Other task"}])
    assert "default_res" in res_def[0]


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_dispose_rows_in_pass_at_k() -> None:
    """Verify that Pass@K runs handle sqe_dispose harness rows with canonical refusals."""
    row_dag = GoldenRow(
        id="sqe-hard-dag-001",
        slice="hard",
        bucket="dag",
        ood=False,
        grader="exact",
        allow_semantic=False,
        harness_id="sqe_dispose",
        expected_tools=["sqe_constraint_solver", "final_answer"],
        prompt="Resolve dependencies",
        expected="checkout compile lint test package",
    )
    row_ood = GoldenRow(
        id="sqe-ood-cycle-001",
        slice="ood",
        bucket="cycle",
        ood=True,
        grader="exact",
        allow_semantic=False,
        harness_id="sqe_dispose",
        expected_tools=["sqe_constraint_solver", "final_answer"],
        prompt="Circular dependency",
        expected="BLOCKED:CYCLE_DETECTED",
    )
    backend = EchoBackend(
        {
            "Resolve dependencies": '{"tool": "final_answer", "args": {"text": "checkout compile lint test package"}}',
            "Circular dependency": '{"tool": "final_answer", "args": {"text": "BLOCKED:CYCLE_DETECTED"}}',
        }
    )
    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    runtime = AgentRuntime(backend, spec=load_spec("sqe_dispose"), teacher=False)
    res = evaluate_pass_at_k(runtime, [row_dag, row_ood], n=2, k=1, harness_id="sqe_dispose")
    assert res["pass_at_k"]["1"] == 1.0
    assert res["ood_synthetic_success_violations"] == 0
    assert res["exact_only"] is True
    assert res["semantic_counted"] is False


@pytest.mark.unit
@pytest.mark.harness
def test_security_scanner_and_data_governance() -> None:
    """SecurityScanner scans code and trajectories; PIIScrubber validates missing presidio."""
    from enhanced_system.harness.data_governance import PIIScrubber
    from enhanced_system.harness.security import SecurityScanner
    from enhanced_system.harness.types import Step, Trajectory

    scanner = SecurityScanner(fail_on_high=False)
    assert scanner.scan_python_code("") == []

    # Clean code
    clean_findings = scanner.scan_python_code("x = 1 + 2\n")
    assert clean_findings == []

    # Trajectory with code action
    traj = Trajectory(
        steps=[
            Step(action=json.dumps({"tool": "code_eval", "args": {"code": "y = 42\n"}})),
            Step(action="not a json"),
        ]
    )
    findings = scanner.scan_trajectory(traj)
    assert findings == []

    # PIIScrubber missing presidio check
    import enhanced_system.harness.data_governance as dg

    if not dg.HAS_PRESIDIO:
        with pytest.raises(ImportError, match="Presidio libraries not found"):
            PIIScrubber()
    else:
        scrubber = PIIScrubber()
        assert scrubber.entities is not None
