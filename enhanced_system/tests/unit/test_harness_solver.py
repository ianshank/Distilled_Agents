"""Unit proofs and contract tests for SqeConstraintSolverTool.

Proves:
- Lexicographically least topological order on valid DAGs
- All 6 reject codes:
    CYCLE_DETECTED (return ok: False, status: "UNSAT")
    UNSAT (return ok: False, status: "UNSAT")
    SCHEMA_VIOLATION (raise ValueError("SCHEMA_VIOLATION: ..."))
    SYNTAX_INVALID (raise ValueError("SYNTAX_INVALID: ..."))
    UNSUPPORTED_THEORY (raise ValueError("UNSUPPORTED_THEORY: ..."))
    RESOURCE_LIMIT (raise ValueError("RESOURCE_LIMIT: ..."))
- No exec, eval, subprocess, network in solver module
- sqe_dispose harness isolation and fail-closed BLOCKED mapping
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from enhanced_system.harness import HarnessFactory
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.tools import (
    TOOL_REGISTRY,
    get_tool,
)
from enhanced_system.harness.tools.solver import SqeConstraintSolverTool


@pytest.mark.unit
@pytest.mark.harness
def test_solver_registry_and_lookup():
    """Verify primary tool id sqe_constraint_solver is registered in TOOL_REGISTRY."""
    assert "sqe_constraint_solver" in TOOL_REGISTRY
    tool = get_tool("sqe_constraint_solver")
    assert isinstance(tool, SqeConstraintSolverTool)
    assert tool.tool_id == "sqe_constraint_solver"

    with pytest.raises(KeyError, match="unknown tool id"):
        get_tool("unknown_constraint_solver_xyz")


@pytest.mark.unit
@pytest.mark.harness
def test_no_forbidden_constructs_in_solver():
    """Verify solver contains no exec, eval, subprocess, network, or external solvers."""
    solver_path = Path(__file__).resolve().parents[2] / "harness" / "tools" / "solver.py"
    source = solver_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_calls = {"exec", "eval", "system", "popen", "spawn"}
    forbidden_modules = {"subprocess", "os", "sys", "socket", "urllib", "requests", "z3", "clingo"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                pytest.fail(f"Forbidden call found in solver: {node.func.id}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in forbidden_modules:
                    pytest.fail(f"Forbidden module import in solver: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in forbidden_modules:
                    pytest.fail(f"Forbidden module importFrom in solver: {node.module}")


@pytest.mark.unit
@pytest.mark.harness
def test_happy_path_sat_lexicographically_least_order():
    """SAT DAG solve returns the lexicographically least valid topological order."""
    tool = SqeConstraintSolverTool()

    # Case 1: Diamond DAG with tie-breaking
    # A precedes B and C; B and C precede D.
    # Valid orders: ["A", "B", "C", "D"] and ["A", "C", "B", "D"].
    # Lexicographically least: ["A", "B", "C", "D"].
    payload = {
        "mode": "solve",
        "graph": {
            "nodes": ["D", "C", "B", "A"],
            "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]],
        },
    }
    raw = tool.run(payload)
    res = json.loads(raw)
    assert res["ok"] is True
    assert res["status"] == "SAT"
    assert res["order"] == ["A", "B", "C", "D"]
    assert res["assignment"] is None
    assert res["reject_code"] is None
    assert res["details"] == {}

    # Case 2: Disconnected components sorted lexicographically
    payload_disjoint = {
        "mode": "solve",
        "graph": {
            "nodes": ["z", "b", "a", "m"],
            "edges": [],
        },
    }
    res_disjoint = json.loads(tool.run(payload_disjoint))
    assert res_disjoint["ok"] is True
    assert res_disjoint["status"] == "SAT"
    assert res_disjoint["order"] == ["a", "b", "m", "z"]

    # Case 3: Complex multi-tier DAG
    payload_complex = {
        "mode": "solve",
        "graph": {
            "nodes": ["task_d", "task_c", "task_b", "task_a"],
            "edges": [["task_a", "task_d"], ["task_b", "task_c"]],
        },
    }
    res_complex = json.loads(tool.run(payload_complex))
    assert res_complex["ok"] is True
    assert res_complex["status"] == "SAT"
    # Candidates initially: task_a, task_b -> pick task_a
    # Next candidates: task_b, task_d -> pick task_b
    # Next candidates: task_c, task_d -> pick task_c
    # Next candidate: task_d
    assert res_complex["order"] == ["task_a", "task_b", "task_c", "task_d"]


@pytest.mark.unit
@pytest.mark.harness
def test_happy_path_sat_boolean_condition_tree():
    """SAT boolean tree solves and produces valid truth assignments."""
    tool = SqeConstraintSolverTool()

    # Formula: (A AND B) AND NOT C
    payload = {
        "mode": "solve",
        "constraints": [
            {"id": "c1", "op": "atom", "atom": "A"},
            {"id": "c2", "op": "atom", "atom": "B"},
            {"id": "c3", "op": "atom", "atom": "C"},
            {"id": "c4", "op": "not", "args": ["c3"]},
            {"id": "c5", "op": "and", "args": ["c1", "c2"]},
            {"id": "root", "op": "and", "args": ["c5", "c4"]},
        ],
    }
    res = json.loads(tool.run(payload))
    assert res["ok"] is True
    assert res["status"] == "SAT"
    assert res["assignment"] == {"A": True, "B": True, "C": False}
    assert res["reject_code"] is None

    # Mixed Graph + Boolean SAT
    mixed_payload = {
        "mode": "solve",
        "graph": {
            "nodes": ["step1", "step2"],
            "edges": [["step1", "step2"]],
        },
        "constraints": [
            {"id": "gate", "op": "atom", "atom": "gate_open"},
        ],
    }
    mixed_res = json.loads(tool.run(mixed_payload))
    assert mixed_res["ok"] is True
    assert mixed_res["status"] == "SAT"
    assert mixed_res["order"] == ["step1", "step2"]
    assert mixed_res["assignment"] == {"gate_open": True}


@pytest.mark.unit
@pytest.mark.harness
def test_mode_check_valid():
    """Mode check verifies valid topological order and truth assignments."""
    tool = SqeConstraintSolverTool()

    payload = {
        "mode": "check",
        "graph": {
            "nodes": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"]],
        },
        "order": ["A", "B", "C"],
        "constraints": [
            {"id": "c1", "op": "atom", "atom": "auth_enabled"},
        ],
        "assignment": {"auth_enabled": True},
    }
    res = json.loads(tool.run(payload))
    assert res["ok"] is True
    assert res["status"] == "SAT"
    assert res["order"] == ["A", "B", "C"]
    assert res["assignment"] == {"auth_enabled": True}


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_cycle_detected():
    """Cycle in graph must return ok: false, status: UNSAT, reject_code: CYCLE_DETECTED."""
    tool = SqeConstraintSolverTool()

    # 2-node cycle
    payload_direct = {
        "mode": "solve",
        "graph": {
            "nodes": ["job_a", "job_b"],
            "edges": [["job_a", "job_b"], ["job_b", "job_a"]],
        },
    }
    res = json.loads(tool.run(payload_direct))
    assert res["ok"] is False
    assert res["status"] == "UNSAT"
    assert res["order"] is None
    assert res["assignment"] is None
    assert res["reject_code"] == "CYCLE_DETECTED"
    assert "cycle_nodes" in res["details"]
    assert set(res["details"]["cycle_nodes"]) == {"job_a", "job_b"}

    # Self-loop
    payload_self = {
        "mode": "solve",
        "graph": {
            "nodes": ["job_a"],
            "edges": [["job_a", "job_a"]],
        },
    }
    res_self = json.loads(tool.run(payload_self))
    assert res_self["ok"] is False
    assert res_self["status"] == "UNSAT"
    assert res_self["reject_code"] == "CYCLE_DETECTED"

    # Multi-node cycle in larger graph
    payload_multi = {
        "mode": "solve",
        "graph": {
            "nodes": ["init", "loop_1", "loop_2", "loop_3", "sink"],
            "edges": [
                ["init", "loop_1"],
                ["loop_1", "loop_2"],
                ["loop_2", "loop_3"],
                ["loop_3", "loop_1"],
                ["loop_3", "sink"],
            ],
        },
    }
    res_multi = json.loads(tool.run(payload_multi))
    assert res_multi["ok"] is False
    assert res_multi["status"] == "UNSAT"
    assert res_multi["reject_code"] == "CYCLE_DETECTED"
    assert res_multi["order"] is None


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_unsat():
    """Unsatisfiable constraints return ok: false, status: UNSAT, reject_code: UNSAT."""
    tool = SqeConstraintSolverTool()

    # Contradiction: P and NOT P
    payload_contradiction = {
        "mode": "solve",
        "constraints": [
            {"id": "p", "op": "atom", "atom": "feature_flag"},
            {"id": "not_p", "op": "not", "args": ["p"]},
            {"id": "root", "op": "and", "args": ["p", "not_p"]},
        ],
    }
    res = json.loads(tool.run(payload_contradiction))
    assert res["ok"] is False
    assert res["status"] == "UNSAT"
    assert res["order"] is None
    assert res["assignment"] is None
    assert res["reject_code"] == "UNSAT"
    assert "reason" in res["details"]

    # Contradiction with fixed assignment
    payload_fixed_conflict = {
        "mode": "solve",
        "constraints": [
            {"id": "c1", "op": "atom", "atom": "flag"},
        ],
        "assignment": {"flag": False},
    }
    res_fixed = json.loads(tool.run(payload_fixed_conflict))
    assert res_fixed["ok"] is False
    assert res_fixed["status"] == "UNSAT"
    assert res_fixed["reject_code"] == "UNSAT"

    # Mode check: order violates topological constraints
    payload_bad_order = {
        "mode": "check",
        "graph": {
            "nodes": ["stage_a", "stage_b"],
            "edges": [["stage_a", "stage_b"]],
        },
        "order": ["stage_b", "stage_a"],
    }
    res_bad_order = json.loads(tool.run(payload_bad_order))
    assert res_bad_order["ok"] is False
    assert res_bad_order["status"] == "UNSAT"
    assert res_bad_order["reject_code"] == "UNSAT"

    # Mode check: assignment violates constraint
    payload_unsat_check = {
        "mode": "check",
        "constraints": [
            {"id": "req", "op": "atom", "atom": "authorized"},
        ],
        "assignment": {"authorized": False},
    }
    res_unsat_check = json.loads(tool.run(payload_unsat_check))
    assert res_unsat_check["ok"] is False
    assert res_unsat_check["status"] == "UNSAT"
    assert res_unsat_check["reject_code"] == "UNSAT"


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_schema_violation():
    """Malformed schema or limit breaches raise ValueError('SCHEMA_VIOLATION: ...')."""
    tool = SqeConstraintSolverTool()

    # Invalid JSON string payload
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: invalid json payload"):
        tool.run("{invalid json")

    # Non-dict payload
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: payload must be a JSON object"):
        tool.run('["not", "an", "object"]')

    # Invalid mode
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: invalid mode 'optimize'"):
        tool.run({"mode": "optimize", "graph": {"nodes": ["A"]}})

    # Empty payload missing both graph and constraints
    with pytest.raises(
        ValueError,
        match="^SCHEMA_VIOLATION: payload must contain at least 'graph' or 'constraints'",
    ):
        tool.run({})

    # Graph not an object
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: graph must be an object"):
        tool.run({"graph": "nodes_a_b"})

    # Nodes not a list
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: graph.nodes must be a list"):
        tool.run({"graph": {"nodes": "single_node"}})

    # Edges not pairs
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: each edge must be a \\[from, to\\] pair"
    ):
        tool.run({"graph": {"nodes": ["A", "B"], "edges": [["A", "B", "C"]]}})

    # Edge references unknown node
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: edge references node 'Z' not in graph.nodes"
    ):
        tool.run({"graph": {"nodes": ["A", "B"], "edges": [["A", "Z"]]}})

    # Input limit breach: max_nodes
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: node count 3 exceeds max_nodes limit 2"
    ):
        tool.run(
            {
                "graph": {"nodes": ["A", "B", "C"]},
                "limits": {"max_nodes": 2},
            }
        )

    # Input limit breach: max_edges
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: edge count 2 exceeds max_edges limit 1"
    ):
        tool.run(
            {
                "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]},
                "limits": {"max_edges": 1},
            }
        )

    # Input limit breach: max_constraints
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: constraint count 2 exceeds limit 1"):
        tool.run(
            {
                "constraints": [
                    {"op": "atom", "atom": "A"},
                    {"op": "atom", "atom": "B"},
                ],
                "limits": {"max_constraints": 1},
            }
        )

    # Missing atom assignment in check mode
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: missing assignment for atom 'P'"):
        tool.run(
            {
                "mode": "check",
                "constraints": [{"op": "atom", "atom": "P"}],
                "assignment": {},
            }
        )


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_syntax_invalid():
    """Unparseable constraint tree or invalid atom syntax raises ValueError('SYNTAX_INVALID: ...')."""
    tool = SqeConstraintSolverTool()

    # Atom missing atom field
    with pytest.raises(
        ValueError, match="^SYNTAX_INVALID: atom constraint at index 0 missing 'atom' string"
    ):
        tool.run({"constraints": [{"op": "atom"}]})

    # Invalid atom identifier syntax (contains operators or whitespace)
    with pytest.raises(ValueError, match="^SYNTAX_INVALID: invalid atom identifier syntax"):
        tool.run({"constraints": [{"op": "atom", "atom": "A & B ()"}]})

    # Atom constraint cannot have args
    with pytest.raises(
        ValueError, match="^SYNTAX_INVALID: atom constraint at index 0 cannot have args"
    ):
        tool.run({"constraints": [{"op": "atom", "atom": "valid_atom", "args": ["something"]}]})

    # 'not' missing arguments
    with pytest.raises(
        ValueError, match="^SYNTAX_INVALID: 'not' operator at index 0 requires args or atom"
    ):
        tool.run({"constraints": [{"op": "not"}]})

    # 'not' with multiple arguments
    with pytest.raises(
        ValueError,
        match="^SYNTAX_INVALID: 'not' operator at index 0 must have exactly one argument",
    ):
        tool.run({"constraints": [{"op": "not", "args": ["A", "B"]}]})

    # 'and' with empty args
    with pytest.raises(
        ValueError, match="^SYNTAX_INVALID: 'and' operator at index 0 requires non-empty args"
    ):
        tool.run({"constraints": [{"op": "and", "args": []}]})

    # Undefined constraint reference
    with pytest.raises(
        ValueError, match="^SYNTAX_INVALID: constraint references undefined id 'nonexistent_id'"
    ):
        tool.run(
            {
                "constraints": [
                    {"id": "root", "op": "and", "args": ["nonexistent_id"]},
                ]
            }
        )

    # Cyclic constraint definitions
    with pytest.raises(ValueError, match="^SYNTAX_INVALID: cyclic constraint definition detected"):
        tool.run(
            {
                "constraints": [
                    {"id": "c1", "op": "not", "args": ["c2"]},
                    {"id": "c2", "op": "not", "args": ["c1"]},
                ]
            }
        )


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_unsupported_theory():
    """Out-of-fragment operators or theories raise ValueError('UNSUPPORTED_THEORY: ...')."""
    tool = SqeConstraintSolverTool()

    # Arithmetic / ILP theory
    with pytest.raises(
        ValueError, match="^UNSUPPORTED_THEORY: operator 'ilp' is outside supported"
    ):
        tool.run({"constraints": [{"op": "ilp", "args": ["x", "y"]}]})

    # SMT / Comparison theory
    with pytest.raises(ValueError, match="^UNSUPPORTED_THEORY: operator '>=' is outside supported"):
        tool.run({"constraints": [{"op": ">=", "args": ["x", 10]}]})

    # Quantifier theory
    with pytest.raises(
        ValueError, match="^UNSUPPORTED_THEORY: operator 'forall' is outside supported"
    ):
        tool.run({"constraints": [{"op": "forall", "args": ["x"]}]})

    # Unsupported graph theory: weights
    with pytest.raises(
        ValueError, match="^UNSUPPORTED_THEORY: graph property 'weights' outside supported"
    ):
        tool.run({"graph": {"nodes": ["A", "B"], "edges": [["A", "B"]], "weights": [1.5]}})

    # Unsupported graph theory: hypergraph
    with pytest.raises(
        ValueError, match="^UNSUPPORTED_THEORY: graph property 'hypergraph' outside supported"
    ):
        tool.run({"graph": {"nodes": ["A", "B"], "hypergraph": True}})


@pytest.mark.unit
@pytest.mark.harness
def test_reject_code_resource_limit():
    """Execution operation or recursion depth cap raises ValueError('RESOURCE_LIMIT: ...')."""
    tool = SqeConstraintSolverTool()

    # Operation cap exceeded during topological sort or SAT search
    with pytest.raises(
        ValueError, match="^RESOURCE_LIMIT: execution operation cap \\(2\\) exceeded"
    ):
        tool.run(
            {
                "graph": {
                    "nodes": ["A", "B", "C", "D"],
                    "edges": [["A", "B"], ["B", "C"], ["C", "D"]],
                },
                "limits": {"max_operations": 2},
            }
        )

    # Inferred node count check after deduplication
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: node count 3 exceeds max_nodes limit 2"
    ):
        tool.run(
            {
                "graph": {
                    "edges": [["A", "B"], ["B", "C"]],
                },
                "limits": {"max_nodes": 2},
            }
        )

    # Boolean not accepted as integer limit
    with pytest.raises(
        ValueError, match="^SCHEMA_VIOLATION: limit 'max_nodes' must be a positive integer"
    ):
        tool.run(
            {
                "graph": {"nodes": ["A"]},
                "limits": {"max_nodes": True},
            }
        )

    # Recursion depth cap exceeded during boolean tree evaluation
    nested_node: dict[str, Any] = {"op": "atom", "atom": "leaf"}
    for _ in range(10):
        nested_node = {"op": "not", "args": [nested_node]}

    with pytest.raises(
        ValueError, match="^RESOURCE_LIMIT: maximum recursion depth \\(3\\) exceeded"
    ):
        tool.run(
            {
                "constraints": [nested_node],
                "limits": {"max_depth": 3},
            }
        )


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_dispose_harness_execution_and_isolation():
    """sqe_dispose harness allowlists sqe_constraint_solver and final_answer only."""
    # sqe_dispose harness allows sqe_constraint_solver
    runtime_dispose = HarnessFactory.create({"harness_id": "sqe_dispose"})
    assert set(runtime_dispose.spec.action.tool_ids) == {"sqe_constraint_solver", "final_answer"}

    # sqe_validate harness is checklist-only
    runtime_validate = HarnessFactory.create({"harness_id": "sqe_validate"})
    assert set(runtime_validate.spec.action.tool_ids) == {"sqe_checklist", "final_answer"}

    # Proposing sqe_checklist on sqe_dispose must fail with DispatchError
    backend_bad_tool = EchoBackend(
        scripted=[
            '{"tool": "sqe_checklist", "args": {"assertions": ["assert 1 == 1"]}}',
            '{"tool": "final_answer", "args": {"text": "done"}}',
        ]
    )
    runtime_bad = HarnessFactory.create({"harness_id": "sqe_dispose", "backend": backend_bad_tool})
    res_bad = runtime_bad.run("Run invalid tool", harness_id="sqe_dispose")
    assert any(step.fault == "parse_error" for step in res_bad.trajectory.steps)

    # Proposing sqe_constraint_solver on sqe_validate must fail with DispatchError
    backend_wrong_harness = EchoBackend(
        scripted=[
            '{"tool": "sqe_constraint_solver", "args": {"graph": {"nodes": ["A"]}}}',
            '{"tool": "final_answer", "args": {"text": "done"}}',
        ]
    )
    runtime_wrong = HarnessFactory.create(
        {"harness_id": "sqe_validate", "backend": backend_wrong_harness}
    )
    res_wrong = runtime_wrong.run("Run solver on validate harness", harness_id="sqe_validate")
    assert any(step.fault == "parse_error" for step in res_wrong.trajectory.steps)

    # Successful solver execution in sqe_dispose
    backend_success = EchoBackend(
        scripted=[
            '{"tool": "sqe_constraint_solver", "args": {"mode": "solve", "graph": {"nodes": ["b", "a"], "edges": [["a", "b"]]}}}',
            '{"tool": "final_answer", "args": {"text": "a b"}}',
        ]
    )
    runtime_success = HarnessFactory.create(
        {"harness_id": "sqe_dispose", "backend": backend_success}
    )
    res_success = runtime_success.run("Sort nodes", harness_id="sqe_dispose")
    assert res_success.final_answer == "a b"
    obs_step = res_success.trajectory.steps[0]
    assert obs_step.tool_id == "sqe_constraint_solver"
    obs_data = json.loads(obs_step.observation)
    assert obs_data["ok"] is True
    assert obs_data["order"] == ["a", "b"]

    # Refusal path with fault token 'blocked'
    backend_blocked = EchoBackend(
        scripted=[
            '{"tool": "sqe_constraint_solver", "args": {"mode": "solve", "graph": {"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]}}}',
            '{"tool": "final_answer", "args": {"text": "BLOCKED:CYCLE_DETECTED"}}',
        ]
    )
    runtime_blocked = HarnessFactory.create(
        {"harness_id": "sqe_dispose", "backend": backend_blocked}
    )
    res_blocked = runtime_blocked.run("Handle cyclic graph", harness_id="sqe_dispose")
    assert res_blocked.final_answer == "BLOCKED:CYCLE_DETECTED"
    assert "blocked" in res_blocked.trajectory.faults
