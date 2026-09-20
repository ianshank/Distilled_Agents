"""Pure-Python symbolic disposition and constraint checking tools."""

from __future__ import annotations

import heapq
import json
import logging
import re
from typing import Any, Optional

from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)

# Standard reject codes conforming to openspec/changes/_shared/blocked-reject-codes.md
CYCLE_DETECTED = "CYCLE_DETECTED"
UNSAT = "UNSAT"
SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
SYNTAX_INVALID = "SYNTAX_INVALID"
UNSUPPORTED_THEORY = "UNSUPPORTED_THEORY"
RESOURCE_LIMIT = "RESOURCE_LIMIT"

STANDARD_REJECT_CODES = frozenset(
    {
        CYCLE_DETECTED,
        UNSAT,
        SCHEMA_VIOLATION,
        SYNTAX_INVALID,
        UNSUPPORTED_THEORY,
        RESOURCE_LIMIT,
    }
)


def _topo_sort(nodes: list[str], edges: list[list[str]]) -> tuple[bool, Optional[list[str]]]:
    """Compute lexicographically least valid topological sort using Kahn's algorithm with a min-heap."""
    in_degree: dict[str, int] = {node: 0 for node in nodes}
    adj: dict[str, list[str]] = {node: [] for node in nodes}

    for edge in edges:
        u, v = edge[0], edge[1]
        adj[u].append(v)
        in_degree[v] += 1

    heap = [node for node in nodes if in_degree[node] == 0]
    heapq.heapify(heap)

    order: list[str] = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in sorted(adj[u]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)

    if len(order) < len(nodes):
        return False, None
    return True, order


def _eval_constraint_node(node: dict[str, Any], assignment: dict[str, bool]) -> bool:
    """Evaluate a boolean constraint tree node under a given truth assignment."""
    if not isinstance(node, dict):
        raise ValueError(f"{SYNTAX_INVALID}: constraint node must be a dict: {node!r}")
    op = node.get("op")
    if not op or not isinstance(op, str):
        raise ValueError(f"{SYNTAX_INVALID}: constraint node missing 'op'")
    op = op.lower().strip()

    if op == "atom":
        atom = node.get("atom")
        if not atom or not isinstance(atom, str):
            raise ValueError(f"{SYNTAX_INVALID}: atom node missing 'atom' string")
        if atom not in assignment:
            raise ValueError(f"{UNSAT}: unassigned atom '{atom}'")
        return bool(assignment[atom])

    if op == "not":
        args = node.get("args")
        if args is None and "arg" in node:
            args = [node["arg"]]
        if not isinstance(args, list) or len(args) != 1:
            raise ValueError(f"{SYNTAX_INVALID}: 'not' operator requires exactly 1 arg")
        return not _eval_constraint_node(args[0], assignment)

    if op == "and":
        args = node.get("args")
        if not isinstance(args, list) or not args:
            raise ValueError(f"{SYNTAX_INVALID}: 'and' operator requires non-empty args list")
        return all(_eval_constraint_node(child, assignment) for child in args)

    if op == "or":
        args = node.get("args")
        if not isinstance(args, list) or not args:
            raise ValueError(f"{SYNTAX_INVALID}: 'or' operator requires non-empty args list")
        return any(_eval_constraint_node(child, assignment) for child in args)

    raise ValueError(f"{UNSUPPORTED_THEORY}: unsupported constraint operator '{op}'")


def _collect_atoms(node: dict[str, Any], atoms: set[str]) -> None:
    """Recursively collect atom identifiers from a boolean constraint tree."""
    if not isinstance(node, dict):
        return
    op = str(node.get("op", "")).lower().strip()
    if op == "atom":
        atom = node.get("atom")
        if atom and isinstance(atom, str):
            atoms.add(atom)
    elif op in {"and", "or"}:
        for child in node.get("args") or []:
            if isinstance(child, dict):
                _collect_atoms(child, atoms)
    elif op == "not":
        args = node.get("args") or ([node["arg"]] if "arg" in node else [])
        for child in args:
            if isinstance(child, dict):
                _collect_atoms(child, atoms)


class ConstraintCheckTool:
    """Validate structured QC schemas, required keys, and predicate constraints (pure Python)."""

    tool_id = "constraint_check"

    def run(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            raise ValueError(f"{SCHEMA_VIOLATION}: payload must be a dictionary")

        settings = get_settings()

        # Extract target document to inspect
        doc: Any = None
        if "document" in payload:
            doc = payload["document"]
        elif "data" in payload:
            doc = payload["data"]
        elif "record" in payload:
            doc = payload["record"]
        elif "target" in payload:
            doc = payload["target"]
        else:
            control_keys = {
                "required",
                "required_keys",
                "predicates",
                "rules",
                "constraints",
                "mode",
                "limits",
                "graph",
                "assignment",
            }
            residual = {k: v for k, v in payload.items() if k not in control_keys}
            if residual:
                doc = residual
            else:
                doc = {}

        if isinstance(doc, str):
            try:
                doc = json.loads(doc)
            except json.JSONDecodeError as exc:
                logger.info("constraint_check schema violation: invalid json string: %s", exc)
                raise ValueError(f"{SCHEMA_VIOLATION}: invalid json document: {exc}") from exc

        if not isinstance(doc, dict):
            logger.info("constraint_check schema violation: document must be an object")
            raise ValueError(f"{SCHEMA_VIOLATION}: document must be an object")

        if len(doc) > 256:
            logger.info("constraint_check resource limit: document has %d keys", len(doc))
            raise ValueError(f"{RESOURCE_LIMIT}: document exceeds maximum key limit (256)")

        # Required key validation
        required = payload.get("required") or payload.get("required_keys") or []
        if not isinstance(required, list):
            raise ValueError(f"{SCHEMA_VIOLATION}: required must be a list of keys")
        missing = [key for key in required if not isinstance(key, str) or key not in doc]
        if missing:
            logger.info("constraint_check violation: missing required keys: %s", missing)
            raise ValueError(f"{SCHEMA_VIOLATION}: missing required keys: {missing}")

        # Predicate validation
        predicates = (
            payload.get("predicates") or payload.get("constraints") or payload.get("rules") or []
        )
        if not isinstance(predicates, list):
            raise ValueError(f"{SCHEMA_VIOLATION}: predicates must be a list")

        if len(predicates) > settings.constraint_max_constraints:
            logger.info("constraint_check resource limit: %d predicates", len(predicates))
            raise ValueError(
                f"{RESOURCE_LIMIT}: predicate count {len(predicates)} exceeds limit {settings.constraint_max_constraints}"
            )

        for pred in predicates:
            if not isinstance(pred, dict):
                raise ValueError(f"{SYNTAX_INVALID}: predicate must be a dictionary")
            key = pred.get("key") or pred.get("field")
            if not key or not isinstance(key, str):
                raise ValueError(f"{SYNTAX_INVALID}: predicate missing 'key' string")
            op = pred.get("op", "==")
            if not isinstance(op, str):
                raise ValueError(f"{SYNTAX_INVALID}: predicate 'op' must be a string")
            op_norm = op.lower().strip()

            # Disallowed theories / execution sandbox traps
            if op_norm in {
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
            }:
                logger.info("constraint_check unsupported theory op: %s", op_norm)
                raise ValueError(f"{UNSUPPORTED_THEORY}: operation '{op}' is not supported")

            val = pred.get("value")
            actual = doc.get(key)
            satisfied = True
            reason = ""

            if op_norm in {"==", "eq"}:
                if actual != val:
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} != expected {val!r}"
            elif op_norm in {"!=", "ne", "neq"}:
                if actual == val:
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} == disallowed {val!r}"
            elif op_norm in {"<", "lt"}:
                if actual is None or not (actual < val):
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} not < {val!r}"
            elif op_norm in {"<=", "le", "lte"}:
                if actual is None or not (actual <= val):
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} not <= {val!r}"
            elif op_norm in {">", "gt"}:
                if actual is None or not (actual > val):
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} not > {val!r}"
            elif op_norm in {">=", "ge", "gte"}:
                if actual is None or not (actual >= val):
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} not >= {val!r}"
            elif op_norm == "in":
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(f"{SYNTAX_INVALID}: 'in' operator requires a list of values")
                if actual not in val:
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} not in {val!r}"
            elif op_norm in {"not_in", "nin"}:
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(
                        f"{SYNTAX_INVALID}: 'not_in' operator requires a list of values"
                    )
                if actual in val:
                    satisfied = False
                    reason = f"field '{key}' with value {actual!r} in disallowed {val!r}"
            elif op_norm == "contains":
                if actual is None or val not in actual:
                    satisfied = False
                    reason = f"field '{key}' does not contain {val!r}"
            elif op_norm == "not_empty":
                if not actual:
                    satisfied = False
                    reason = f"field '{key}' is empty or null"
            elif op_norm == "exists":
                if key not in doc:
                    satisfied = False
                    reason = f"field '{key}' does not exist in document"
            elif op_norm in {"type", "is_type"}:
                type_map: dict[str, Any] = {
                    "str": str,
                    "string": str,
                    "int": int,
                    "integer": int,
                    "float": float,
                    "number": (int, float),
                    "bool": bool,
                    "boolean": bool,
                    "list": list,
                    "array": list,
                    "dict": dict,
                    "object": dict,
                }
                expected_type = type_map.get(str(val).lower())
                if expected_type is None:
                    raise ValueError(f"{SYNTAX_INVALID}: unknown type '{val}' in type check")
                if expected_type in (int, (int, float)) and isinstance(actual, bool):
                    satisfied = False
                    reason = f"field '{key}' is bool, expected int/number"
                elif not isinstance(actual, expected_type):
                    satisfied = False
                    reason = f"field '{key}' has type {type(actual).__name__}, expected {val}"
            elif op_norm in {"regex", "matches"}:
                if not isinstance(val, str):
                    raise ValueError(f"{SYNTAX_INVALID}: regex pattern must be a string")
                if actual is None or not re.search(val, str(actual)):
                    satisfied = False
                    reason = f"field '{key}' value {actual!r} does not match pattern {val!r}"
            else:
                raise ValueError(f"{UNSUPPORTED_THEORY}: operator '{op}' is not supported")

            if not satisfied:
                logger.info("constraint_check violation: %s", reason)
                raise ValueError(f"{UNSAT}: {reason}")

        # Check graph if present in payload
        if "graph" in payload:
            graph_data = payload["graph"]
            if not isinstance(graph_data, dict):
                raise ValueError(f"{SCHEMA_VIOLATION}: graph must be an object")
            nodes = graph_data.get("nodes") or []
            edges = graph_data.get("edges") or []
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError(f"{SCHEMA_VIOLATION}: graph nodes and edges must be lists")
            ok, order = _topo_sort(nodes, edges)
            if not ok:
                logger.info("constraint_check cycle violation in graph")
                raise ValueError(f"{CYCLE_DETECTED}: dependency graph contains directed cycle")

        logger.info(
            "constraint_check success: verified %d keys and %d predicates",
            len(doc),
            len(predicates),
        )
        return json.dumps(
            {
                "ok": True,
                "status": "SAT",
                "checked_keys": sorted(doc.keys()),
                "predicates_checked": len(predicates),
            }
        )


class SqeConstraintSolverTool:
    """Pure-Python DAG topological ordering and boolean constraint tree solver."""

    tool_id = "sqe_constraint_solver"

    def run(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            raise ValueError(f"{SCHEMA_VIOLATION}: payload must be a dictionary")

        settings = get_settings()

        mode = payload.get("mode", "solve")
        if mode not in {"check", "solve"}:
            raise ValueError(f"{SCHEMA_VIOLATION}: mode must be 'check' or 'solve'")

        limits = payload.get("limits") or {}
        if not isinstance(limits, dict):
            raise ValueError(f"{SCHEMA_VIOLATION}: limits must be a dictionary")

        max_nodes = limits.get("max_nodes", settings.constraint_max_nodes)
        max_edges = limits.get("max_edges", settings.constraint_max_edges)
        max_constraints = limits.get("max_constraints", settings.constraint_max_constraints)

        order: Optional[list[str]] = None
        if "graph" in payload:
            graph = payload["graph"]
            if not isinstance(graph, dict):
                raise ValueError(f"{SCHEMA_VIOLATION}: graph must be an object")
            nodes = graph.get("nodes") or []
            edges = graph.get("edges") or []
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError(f"{SCHEMA_VIOLATION}: graph nodes and edges must be lists")

            if len(nodes) > max_nodes:
                raise ValueError(
                    f"{RESOURCE_LIMIT}: node count {len(nodes)} exceeds max_nodes {max_nodes}"
                )
            if len(edges) > max_edges:
                raise ValueError(
                    f"{RESOURCE_LIMIT}: edge count {len(edges)} exceeds max_edges {max_edges}"
                )

            for node in nodes:
                if not isinstance(node, str) or not node.strip():
                    raise ValueError(f"{SCHEMA_VIOLATION}: graph nodes must be non-empty strings")

            nodes_set = set(nodes)
            clean_edges: list[list[str]] = []
            for edge in edges:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValueError(f"{SCHEMA_VIOLATION}: each edge must be a [from, to] pair")
                u, v = str(edge[0]), str(edge[1])
                if u not in nodes_set or v not in nodes_set:
                    raise ValueError(
                        f"{SCHEMA_VIOLATION}: edge endpoint ({u}, {v}) not in graph nodes"
                    )
                clean_edges.append([u, v])

            ok_topo, topo_order = _topo_sort(nodes, clean_edges)
            if not ok_topo:
                logger.info("sqe_constraint_solver: cycle detected in graph")
                return json.dumps(
                    {
                        "ok": False,
                        "status": "UNSAT",
                        "order": None,
                        "assignment": None,
                        "reject_code": CYCLE_DETECTED,
                        "details": {"reason": "cycle detected in graph"},
                    }
                )
            order = topo_order

        # Handle constraints tree
        sat_assignment: Optional[dict[str, bool]] = None
        if "constraints" in payload:
            raw_constraints = payload["constraints"]
            constraint_list: list[dict[str, Any]] = []
            if isinstance(raw_constraints, list):
                constraint_list = raw_constraints
            elif isinstance(raw_constraints, dict):
                constraint_list = [raw_constraints]
            else:
                raise ValueError(f"{SCHEMA_VIOLATION}: constraints must be a list or object")

            if len(constraint_list) > max_constraints:
                raise ValueError(
                    f"{RESOURCE_LIMIT}: constraint count {len(constraint_list)} exceeds max_constraints {max_constraints}"
                )

            assignment_input = payload.get("assignment")
            if assignment_input is not None and not isinstance(assignment_input, dict):
                raise ValueError(f"{SCHEMA_VIOLATION}: assignment must be a dictionary")

            # Validate node ops before solving
            atoms: set[str] = set()
            for c_node in constraint_list:
                _collect_atoms(c_node, atoms)

            if assignment_input is not None:
                # Validate against provided assignment
                try:
                    all_sat = all(
                        _eval_constraint_node(c_node, assignment_input)
                        for c_node in constraint_list
                    )
                except ValueError as exc:
                    if str(exc).startswith(f"{UNSUPPORTED_THEORY}:"):
                        raise
                    if str(exc).startswith(f"{SYNTAX_INVALID}:"):
                        raise
                    all_sat = False

                if not all_sat:
                    logger.info("sqe_constraint_solver: constraints unsatisfiable under assignment")
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "UNSAT",
                            "order": None,
                            "assignment": None,
                            "reject_code": UNSAT,
                            "details": {
                                "reason": "constraints evaluated to false under assignment"
                            },
                        }
                    )
                sat_assignment = {k: bool(v) for k, v in assignment_input.items()}
            else:
                # Solve for a satisfying assignment
                free_atoms = sorted(atoms)
                if len(free_atoms) > 16:
                    raise ValueError(
                        f"{RESOURCE_LIMIT}: too many free boolean atoms ({len(free_atoms)})"
                    )

                found = False
                for i in range(1 << len(free_atoms)):
                    candidate = {
                        atom: bool((i >> bit_idx) & 1) for bit_idx, atom in enumerate(free_atoms)
                    }
                    try:
                        if all(
                            _eval_constraint_node(c_node, candidate) for c_node in constraint_list
                        ):
                            sat_assignment = candidate
                            found = True
                            break
                    except ValueError as exc:
                        if str(exc).startswith(f"{UNSUPPORTED_THEORY}:"):
                            raise
                        if str(exc).startswith(f"{SYNTAX_INVALID}:"):
                            raise

                if not found:
                    logger.info("sqe_constraint_solver: contradictory constraints (UNSAT)")
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "UNSAT",
                            "order": None,
                            "assignment": None,
                            "reject_code": UNSAT,
                            "details": {"reason": "contradictory constraints"},
                        }
                    )

        logger.info("sqe_constraint_solver: SAT achieved")
        return json.dumps(
            {
                "ok": True,
                "status": "SAT",
                "order": order,
                "assignment": sat_assignment,
                "reject_code": None,
                "details": {},
            }
        )
