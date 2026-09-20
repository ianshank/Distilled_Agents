"""Unit tests for ConstraintCheckTool, SqeConstraintSolverTool, and TOOL_REGISTRY."""

from __future__ import annotations

import json

import pytest
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.harness.tools import (
    TOOL_REGISTRY,
    ConstraintCheckTool,
    get_tool,
)
from enhanced_system.harness.tools.constraint import (
    CYCLE_DETECTED,
    RESOURCE_LIMIT,
    SCHEMA_VIOLATION,
    SYNTAX_INVALID,
    UNSAT,
    UNSUPPORTED_THEORY,
)
from enhanced_system.harness.tools.solver import SqeConstraintSolverTool


@pytest.mark.unit
@pytest.mark.harness
def test_tool_registry_lookup_and_unknown():
    """Registry returns instances for registered tools and KeyError for unknown."""
    tool = get_tool("constraint_check")
    assert isinstance(tool, ConstraintCheckTool)
    assert tool.tool_id == "constraint_check"

    solver = get_tool("sqe_constraint_solver")
    assert isinstance(solver, SqeConstraintSolverTool)
    assert solver.tool_id == "sqe_constraint_solver"

    assert "constraint_check" in TOOL_REGISTRY
    assert "sqe_constraint_solver" in TOOL_REGISTRY

    with pytest.raises(KeyError, match="unknown tool id"):
        get_tool("nonexistent_tool_xyz")


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_valid_document_and_predicates():
    """Valid document with required keys and diverse predicates passes with SAT."""
    tool = ConstraintCheckTool()
    payload = {
        "document": {
            "service": "billing",
            "version": "2.1.0",
            "retries": 3,
            "timeout_seconds": 15.5,
            "active": True,
            "tags": ["prod", "finance"],
            "meta": {"env": "production"},
        },
        "required": ["service", "version", "retries", "active"],
        "predicates": [
            {"key": "service", "op": "==", "value": "billing"},
            {"key": "version", "op": "regex", "value": r"^\d+\.\d+\.\d+$"},
            {"key": "retries", "op": "<=", "value": 5},
            {"key": "retries", "op": ">", "value": 0},
            {"key": "retries", "op": ">=", "value": 3},
            {"key": "retries", "op": "<", "value": 10},
            {"key": "timeout_seconds", "op": "!=", "value": 0.0},
            {"key": "active", "op": "==", "value": True},
            {"key": "tags", "op": "contains", "value": "prod"},
            {"key": "tags", "op": "not_empty"},
            {"key": "service", "op": "in", "value": ["auth", "billing", "orders"]},
            {"key": "service", "op": "not_in", "value": ["deprecated", "legacy"]},
            {"key": "service", "op": "exists"},
            {"key": "retries", "op": "type", "value": "int"},
            {"key": "timeout_seconds", "op": "is_type", "value": "number"},
            {"key": "active", "op": "type", "value": "bool"},
            {"key": "tags", "op": "type", "value": "list"},
            {"key": "meta", "op": "type", "value": "dict"},
        ],
    }
    raw_obs = tool.run(payload)
    obs = json.loads(raw_obs)
    assert obs["ok"] is True
    assert obs["status"] == "SAT"
    assert "service" in obs["checked_keys"]
    assert obs["predicates_checked"] == 18


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_json_string_and_alternate_keys():
    """Document passed as JSON string or via 'data'/'record' is parsed correctly."""
    tool = ConstraintCheckTool()
    # document as JSON string
    res1 = json.loads(
        tool.run(
            {
                "document": json.dumps({"app": "gateway", "port": 8080}),
                "required": ["app", "port"],
            }
        )
    )
    assert res1["ok"] is True

    # 'data' key
    res2 = json.loads(tool.run({"data": {"app": "worker"}, "required": ["app"]}))
    assert res2["ok"] is True

    # 'record' key
    res3 = json.loads(tool.run({"record": {"app": "agent"}, "required": ["app"]}))
    assert res3["ok"] is True

    # residual payload
    res4 = json.loads(tool.run({"app": "auth", "required": ["app"]}))
    assert res4["ok"] is True


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_schema_violations():
    """Malformed payload or document structures raise SCHEMA_VIOLATION."""
    tool = ConstraintCheckTool()

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run("not a dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"document": "{invalid json"})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"document": ["not a dict"]})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"document": {"a": 1}, "required": "not a list"})

    with pytest.raises(ValueError, match=f"{SCHEMA_VIOLATION}.*missing required keys"):
        tool.run({"document": {"a": 1}, "required": ["a", "missing_key"]})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"document": {"a": 1}, "predicates": "not a list"})


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_predicate_violations_raise_unsat():
    """Failing predicates raise ValueError prefixed with UNSAT."""
    tool = ConstraintCheckTool()
    doc = {"val": 10, "name": "prod", "empty_list": []}

    with pytest.raises(ValueError, match=f"{UNSAT}.*!="):
        tool.run({"document": doc, "predicates": [{"key": "name", "op": "==", "value": "dev"}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*=="):
        tool.run({"document": doc, "predicates": [{"key": "name", "op": "!=", "value": "prod"}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*not <"):
        tool.run({"document": doc, "predicates": [{"key": "val", "op": "<", "value": 5}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*not <="):
        tool.run({"document": doc, "predicates": [{"key": "val", "op": "<=", "value": 9}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*not >"):
        tool.run({"document": doc, "predicates": [{"key": "val", "op": ">", "value": 20}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*not >="):
        tool.run({"document": doc, "predicates": [{"key": "val", "op": ">=", "value": 15}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*not in"):
        tool.run(
            {
                "document": doc,
                "predicates": [{"key": "name", "op": "in", "value": ["stage", "test"]}],
            }
        )

    with pytest.raises(ValueError, match=f"{UNSAT}.*disallowed"):
        tool.run(
            {"document": doc, "predicates": [{"key": "name", "op": "not_in", "value": ["prod"]}]}
        )

    with pytest.raises(ValueError, match=f"{UNSAT}.*does not contain"):
        tool.run(
            {"document": doc, "predicates": [{"key": "name", "op": "contains", "value": "xyz"}]}
        )

    with pytest.raises(ValueError, match=f"{UNSAT}.*empty or null"):
        tool.run({"document": doc, "predicates": [{"key": "empty_list", "op": "not_empty"}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*does not exist"):
        tool.run({"document": doc, "predicates": [{"key": "nonexistent", "op": "exists"}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*expected int"):
        tool.run({"document": doc, "predicates": [{"key": "name", "op": "type", "value": "int"}]})

    with pytest.raises(ValueError, match=f"{UNSAT}.*does not match pattern"):
        tool.run(
            {"document": doc, "predicates": [{"key": "name", "op": "regex", "value": r"^\d+$"}]}
        )


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_unsupported_theories_and_syntax_invalid():
    """Disallowed ops (exec/eval/smt) or bad syntax raise appropriate codes."""
    tool = ConstraintCheckTool()
    doc = {"x": 1}

    for disallowed in [
        "exec",
        "eval",
        "code",
        "run",
        "subprocess",
        "shell",
        "os",
        "system",
        "ilp",
        "smt",
    ]:
        with pytest.raises(ValueError, match=f"{UNSUPPORTED_THEORY}.*not supported"):
            tool.run({"document": doc, "predicates": [{"key": "x", "op": disallowed}]})

    with pytest.raises(ValueError, match=UNSUPPORTED_THEORY):
        tool.run({"document": doc, "predicates": [{"key": "x", "op": "unknown_future_op"}]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": ["not a dict"]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": [{"op": "=="}]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": [{"key": "x", "op": 123}]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": [{"key": "x", "op": "in", "value": "not a list"}]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": [{"key": "x", "op": "not_in", "value": 123}]})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run(
            {"document": doc, "predicates": [{"key": "x", "op": "type", "value": "unknown_type"}]}
        )

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        tool.run({"document": doc, "predicates": [{"key": "x", "op": "regex", "value": 123}]})


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_resource_limits():
    """Excessive keys or predicates raise RESOURCE_LIMIT."""
    tool = ConstraintCheckTool()
    large_doc = {f"k_{i}": i for i in range(300)}
    with pytest.raises(ValueError, match=RESOURCE_LIMIT):
        tool.run({"document": large_doc})

    many_predicates = [{"key": "x", "op": "==", "value": 1} for _ in range(150)]
    with pytest.raises(ValueError, match=RESOURCE_LIMIT):
        tool.run({"document": {"x": 1}, "predicates": many_predicates})


@pytest.mark.unit
@pytest.mark.harness
def test_constraint_check_graph_support():
    """ConstraintCheckTool verifies graph DAGs and raises on cycles."""
    tool = ConstraintCheckTool()
    # Valid DAG passes
    res = json.loads(
        tool.run(
            {
                "document": {"task": "deploy"},
                "graph": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            }
        )
    )
    assert res["ok"] is True

    # Invalid graph
    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"graph": "not a dict"})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        tool.run({"graph": {"nodes": "not a list", "edges": []}})

    # Cyclic graph raises CYCLE_DETECTED
    with pytest.raises(ValueError, match=f"{CYCLE_DETECTED}.*cycle"):
        tool.run(
            {
                "graph": {"nodes": ["A", "B"], "edges": [["A", "B"], ["B", "A"]]},
            }
        )


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_solver_dag_sat_topological_order():
    """SqeConstraintSolverTool computes lexicographically least topological order."""
    solver = SqeConstraintSolverTool()
    payload = {
        "mode": "solve",
        "graph": {
            "nodes": ["D", "C", "B", "A"],
            "edges": [["B", "D"], ["A", "C"], ["C", "D"]],
        },
    }
    raw = solver.run(payload)
    res = json.loads(raw)
    assert res["ok"] is True
    assert res["status"] == "SAT"
    assert res["reject_code"] is None
    # Topological valid orders with tie-breaking:
    # A has in-degree 0, B has in-degree 0 -> min heap pops A first, then B (or depends on graph)
    # A -> C, B -> D, C -> D
    # At start: A and B in-degree 0 -> pops A. Remaining in-degree 0: B. Pops B.
    # After B popped: C has in-degree 0. Pops C. D in-degree 0 -> pops D.
    assert res["order"] == ["A", "B", "C", "D"]


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_solver_cycle_detection():
    """Cyclic graph returns ok: false, status: UNSAT, reject_code: CYCLE_DETECTED."""
    solver = SqeConstraintSolverTool()
    payload = {
        "mode": "solve",
        "graph": {
            "nodes": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"], ["C", "A"]],
        },
    }
    raw = solver.run(payload)
    res = json.loads(raw)
    assert res["ok"] is False
    assert res["status"] == "UNSAT"
    assert res["reject_code"] == CYCLE_DETECTED
    assert res["order"] is None
    assert "cycle" in res["details"]["reason"]


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_solver_boolean_constraints():
    """Boolean constraint trees evaluate and solve correctly."""
    solver = SqeConstraintSolverTool()

    # Satisfiable with provided assignment
    payload_sat = {
        "constraints": [
            {
                "op": "and",
                "args": [{"op": "atom", "atom": "auth_ok"}, {"op": "atom", "atom": "db_ok"}],
            },
            {"op": "not", "args": [{"op": "atom", "atom": "has_errors"}]},
        ],
        "assignment": {"auth_ok": True, "db_ok": True, "has_errors": False},
    }
    res_sat = json.loads(solver.run(payload_sat))
    assert res_sat["ok"] is True
    assert res_sat["status"] == "SAT"
    assert res_sat["assignment"] == {"auth_ok": True, "db_ok": True, "has_errors": False}

    # Unsatisfiable under assignment
    payload_unsat = {
        "constraints": [
            {"op": "atom", "atom": "auth_ok"},
        ],
        "assignment": {"auth_ok": False},
    }
    res_unsat = json.loads(solver.run(payload_unsat))
    assert res_unsat["ok"] is False
    assert res_unsat["status"] == "UNSAT"
    assert res_unsat["reject_code"] == UNSAT
    assert res_unsat["order"] is None

    # Solve free boolean atoms
    payload_solve = {
        "constraints": [
            {"op": "or", "args": [{"op": "atom", "atom": "p"}, {"op": "atom", "atom": "q"}]},
            {"op": "not", "args": [{"op": "atom", "atom": "p"}]},
        ]
    }
    res_solve = json.loads(solver.run(payload_solve))
    assert res_solve["ok"] is True
    assert res_solve["status"] == "SAT"
    assert res_solve["assignment"]["p"] is False
    assert res_solve["assignment"]["q"] is True

    # Contradictory without assignment
    payload_contradiction = {
        "constraints": [
            {"op": "atom", "atom": "p"},
            {"op": "not", "args": [{"op": "atom", "atom": "p"}]},
        ]
    }
    res_contra = json.loads(solver.run(payload_contradiction))
    assert res_contra["ok"] is False
    assert res_contra["status"] == "UNSAT"
    assert res_contra["reject_code"] == UNSAT


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_solver_schema_and_resource_limits():
    """SqeConstraintSolverTool enforces schema and resource bounds."""
    solver = SqeConstraintSolverTool()

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run("not a dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"mode": "invalid_mode"})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"limits": "not a dict"})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"graph": "not a dict"})

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"graph": {"nodes": [123], "edges": []}})  # type: ignore[list-item]

    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"graph": {"nodes": ["A"], "edges": ["not a pair"]}})  # type: ignore[list-item]

    with pytest.raises(ValueError, match=f"{SCHEMA_VIOLATION}: edge references node"):
        solver.run({"graph": {"nodes": ["A"], "edges": [["A", "UNKNOWN"]]}})

    # Resource limit on nodes
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: node count"):
        solver.run(
            {
                "graph": {"nodes": [f"N_{i}" for i in range(100)], "edges": []},
                "limits": {"max_nodes": 10},
            }
        )

    # Unsupported constraint op
    with pytest.raises(ValueError, match=UNSUPPORTED_THEORY):
        solver.run({"constraints": [{"op": "ilp_solver", "args": []}]})


@pytest.mark.unit
@pytest.mark.harness
def test_falsifier_cyclic_or_unsat_never_reports_sat():
    """Falsifier: Cyclic graph or unsatisfiable constraints MUST NEVER return ok: true or valid order."""
    solver = SqeConstraintSolverTool()

    # Cyclic
    cyclic_payload = {
        "graph": {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"], ["Y", "X"]],
        }
    }
    res_cyclic = json.loads(solver.run(cyclic_payload))
    assert res_cyclic["ok"] is False
    assert res_cyclic["status"] != "SAT"
    assert res_cyclic["order"] is None

    # Unsatisfiable constraints
    unsat_payload = {
        "constraints": [
            {"op": "atom", "atom": "A"},
            {"op": "not", "args": [{"op": "atom", "atom": "A"}]},
        ]
    }
    res_unsat = json.loads(solver.run(unsat_payload))
    assert res_unsat["ok"] is False
    assert res_unsat["status"] != "SAT"
    assert res_unsat["order"] is None

    # ConstraintCheckTool violation never returns SAT
    check_tool = ConstraintCheckTool()
    with pytest.raises(ValueError) as excinfo:
        check_tool.run({"document": {"x": 1}, "predicates": [{"key": "x", "op": "==", "value": 2}]})
    assert UNSAT in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.harness
def test_agent_runtime_with_qc_constraints_harness():
    """AgentRuntime executes qc_constraints harness with constraint_check and final_answer."""
    spec = load_spec("qc_constraints")
    assert "constraint_check" in spec.action.tool_ids
    assert "final_answer" in spec.action.tool_ids

    # Scripted mock backend with successful constraint_check followed by final_answer
    scripted_success = [
        json.dumps(
            {
                "tool": "constraint_check",
                "args": {
                    "document": {"service": "catalog", "healthy": True},
                    "required": ["service", "healthy"],
                    "predicates": [{"key": "healthy", "op": "==", "value": True}],
                },
            }
        ),
        json.dumps({"tool": "final_answer", "args": {"text": "service catalog is healthy"}}),
    ]
    backend = EchoBackend(scripted=scripted_success)
    runtime = AgentRuntime(backend, spec=spec)
    result = runtime.run("Validate catalog service health.")

    assert result.final_answer == "service catalog is healthy"
    assert len(result.trajectory.steps) == 2
    assert result.trajectory.steps[0].tool_id == "constraint_check"
    assert result.trajectory.steps[1].tool_id == "final_answer"
    assert result.trajectory.faults == []

    # Scripted mock backend with violating constraint_check resulting in tool_error and BLOCKED refusal
    scripted_failure = [
        json.dumps(
            {
                "tool": "constraint_check",
                "args": {
                    "document": {"service": "auth", "healthy": False},
                    "required": ["service"],
                    "predicates": [{"key": "healthy", "op": "==", "value": True}],
                },
            }
        ),
        json.dumps({"tool": "final_answer", "args": {"text": "BLOCKED:UNSAT"}}),
    ]
    fail_backend = EchoBackend(scripted=scripted_failure)
    fail_runtime = AgentRuntime(fail_backend, spec=spec)
    fail_result = fail_runtime.run("Validate auth service health.")

    assert fail_result.final_answer == "BLOCKED:UNSAT"
    assert "tool_error" in fail_result.trajectory.faults


@pytest.mark.unit
@pytest.mark.harness
def test_eval_constraint_node_internal_and_syntax_errors():
    """Direct tests for _eval_constraint_node and _collect_atoms edge cases."""
    from enhanced_system.harness.tools.constraint import (
        _collect_atoms,
        _eval_constraint_node,
    )

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node("not a dict", {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node({}, {})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node({"op": "atom"}, {})

    with pytest.raises(ValueError, match=UNSAT):
        _eval_constraint_node({"op": "atom", "atom": "A"}, {})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node({"op": "not", "args": []}, {})

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node({"op": "and", "args": "not a list"}, {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=SYNTAX_INVALID):
        _eval_constraint_node({"op": "or", "args": []}, {})

    with pytest.raises(ValueError, match=UNSUPPORTED_THEORY):
        _eval_constraint_node({"op": "xor", "args": []}, {})

    atoms: set[str] = set()
    _collect_atoms("not a dict", atoms)  # type: ignore[arg-type]
    assert atoms == set()

    tree = {
        "op": "and",
        "args": [
            {"op": "atom", "atom": "A"},
            {
                "op": "or",
                "args": [
                    {"op": "atom", "atom": "B"},
                    {"op": "not", "arg": {"op": "atom", "atom": "C"}},
                ],
            },
        ],
    }
    _collect_atoms(tree, atoms)
    assert atoms == {"A", "B", "C"}


@pytest.mark.unit
@pytest.mark.harness
def test_sqe_solver_edge_cases():
    """Edge cases for sqe_constraint_solver such as single dict constraint and free atoms limit."""
    solver = SqeConstraintSolverTool()

    # Single dict constraint
    res = json.loads(
        solver.run({"constraints": [{"op": "atom", "atom": "flag"}], "assignment": {"flag": True}})
    )
    assert res["ok"] is True
    assert res["status"] == "SAT"

    # Too many free atoms (> 16)
    too_many = [{"op": "atom", "atom": f"p_{i}"} for i in range(20)]
    with pytest.raises(ValueError, match="^SCHEMA_VIOLATION: constraint count"):
        solver.run({"constraints": too_many, "limits": {"max_constraints": 10}})

    # Edge cases in graph edges
    with pytest.raises(ValueError, match=SCHEMA_VIOLATION):
        solver.run({"graph": {"nodes": ["A", "B"], "edges": [["A"]]}})  # type: ignore[list-item]

    # Unsupported theory in constraints with assignment
    with pytest.raises(ValueError, match=UNSUPPORTED_THEORY):
        solver.run({"constraints": [{"op": "smt_formula"}], "assignment": {"x": True}})

    # Unsupported theory in constraints without assignment (during solve)
    with pytest.raises(ValueError, match=UNSUPPORTED_THEORY):
        solver.run({"constraints": [{"op": "smt_formula"}]})
