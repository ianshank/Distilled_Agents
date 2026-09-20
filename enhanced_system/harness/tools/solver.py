"""Deterministic pure-Python constraint solver for DAGs and boolean trees.

Conforms to openspec/changes/symbolic-disposition/ and
openspec/changes/_shared/blocked-reject-codes.md.
"""

from __future__ import annotations

import heapq
import json
import re
from typing import Any, Callable

ATOM_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-\.:]*$")

SUPPORTED_OPS = {"and", "or", "not", "atom"}


class SqeConstraintSolverTool:
    """Pure-Python solver for DAG topological ordering and boolean condition trees.

    Tool ID: sqe_constraint_solver
    Supported operations:
      - DAG topological sorting with lexicographical tie-breaking
      - Propositional boolean condition trees ('and', 'or', 'not', 'atom')
    Rejection behavior:
      - CYCLE_DETECTED, UNSAT: return ok: False, status: "UNSAT", reject_code: <CODE>
      - SCHEMA_VIOLATION, SYNTAX_INVALID, UNSUPPORTED_THEORY, RESOURCE_LIMIT:
        raise ValueError("<CODE>: ...")
    """

    tool_id = "sqe_constraint_solver"

    def run(self, payload: dict[str, Any] | str) -> str:
        """Execute constraint check or solve and return JSON string response."""
        data = self._parse_payload(payload)
        mode = self._parse_mode(data)
        limits = self._parse_limits(data)

        max_nodes = limits["max_nodes"]
        max_edges = limits["max_edges"]
        max_constraints = limits["max_constraints"]
        max_operations = limits["max_operations"]
        max_depth = limits["max_depth"]

        op_counter = [0]

        def record_op(count: int = 1) -> None:
            op_counter[0] += count
            if op_counter[0] > max_operations:
                raise ValueError(
                    f"RESOURCE_LIMIT: execution operation cap ({max_operations}) exceeded"
                )

        has_graph = "graph" in data and data["graph"] is not None
        has_constraints = "constraints" in data and data["constraints"] is not None

        if not has_graph and not has_constraints:
            raise ValueError(
                "SCHEMA_VIOLATION: payload must contain at least 'graph' or 'constraints'"
            )

        nodes: list[str] = []
        edges: list[list[str]] = []
        check_order: list[str] | None = None

        if has_graph:
            nodes, edges = self._parse_graph(data["graph"], max_nodes, max_edges)
            if "order" in data and data["order"] is not None:
                check_order = self._parse_order(data["order"])

        raw_constraints = data.get("constraints")
        parsed_constraints: list[dict[str, Any]] = []
        all_atoms: set[str] = set()

        if has_constraints:
            parsed_constraints, all_atoms = self._parse_constraints(
                raw_constraints, max_constraints, record_op, max_depth
            )

        input_assignment: dict[str, bool] = {}
        if "assignment" in data and data["assignment"] is not None:
            input_assignment = self._parse_assignment(data["assignment"])

        # 1. Process Graph
        topo_order: list[str] | None = None
        if has_graph:
            is_dag, order, cycle_nodes = self._solve_topo(nodes, edges, record_op)
            if not is_dag:
                return json.dumps(
                    {
                        "ok": False,
                        "status": "UNSAT",
                        "order": None,
                        "assignment": None,
                        "reject_code": "CYCLE_DETECTED",
                        "details": {
                            "reason": "cycle detected in dependency graph",
                            "cycle_nodes": cycle_nodes,
                        },
                    }
                )

            if mode == "check" and check_order is not None:
                valid_order, reason = self._verify_order(nodes, edges, check_order, record_op)
                if not valid_order:
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "UNSAT",
                            "order": None,
                            "assignment": None,
                            "reject_code": "UNSAT",
                            "details": {"reason": reason},
                        }
                    )
                topo_order = check_order
            else:
                topo_order = order

        # 2. Process Boolean Constraints
        final_assignment: dict[str, bool] | None = None
        if has_constraints:
            roots = self._find_roots(parsed_constraints)

            if mode == "check":
                for atom in all_atoms:
                    if atom not in input_assignment:
                        raise ValueError(f"SCHEMA_VIOLATION: missing assignment for atom '{atom}'")
                sat = self._eval_all_roots(roots, input_assignment, record_op, max_depth)
                if not sat:
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "UNSAT",
                            "order": None,
                            "assignment": None,
                            "reject_code": "UNSAT",
                            "details": {"reason": "constraints unsatisfied under given assignment"},
                        }
                    )
                final_assignment = {k: input_assignment[k] for k in sorted(input_assignment)}
            else:
                # mode == "solve"
                fixed_assignment = dict(input_assignment)
                free_atoms = sorted(a for a in all_atoms if a not in fixed_assignment)

                sat_assignment = self._solve_sat(
                    roots, free_atoms, fixed_assignment, record_op, max_depth
                )
                if sat_assignment is None:
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "UNSAT",
                            "order": None,
                            "assignment": None,
                            "reject_code": "UNSAT",
                            "details": {"reason": "constraints are unsatisfiable"},
                        }
                    )
                final_assignment = {k: sat_assignment[k] for k in sorted(sat_assignment)}

        return json.dumps(
            {
                "ok": True,
                "status": "SAT",
                "order": topo_order,
                "assignment": final_assignment,
                "reject_code": None,
                "details": {},
            }
        )

    def _parse_payload(self, payload: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"SCHEMA_VIOLATION: invalid json payload: {exc}") from exc
        elif isinstance(payload, dict):
            data = payload
        else:
            raise ValueError("SCHEMA_VIOLATION: payload must be an object")

        if not isinstance(data, dict):
            raise ValueError("SCHEMA_VIOLATION: payload must be a JSON object")
        return data

    def _parse_mode(self, data: dict[str, Any]) -> str:
        mode = data.get("mode", "solve")
        if not isinstance(mode, str):
            raise ValueError("SCHEMA_VIOLATION: mode must be a string")
        mode_lower = mode.strip().lower()
        if mode_lower not in ("check", "solve"):
            raise ValueError(f"SCHEMA_VIOLATION: invalid mode '{mode}', must be 'check' or 'solve'")
        return mode_lower

    def _parse_limits(self, data: dict[str, Any]) -> dict[str, int]:
        raw_limits = data.get("limits") or {}
        if not isinstance(raw_limits, dict):
            raise ValueError("SCHEMA_VIOLATION: limits must be an object")

        for k, v in raw_limits.items():
            if type(v) is not int or v <= 0:
                raise ValueError(f"SCHEMA_VIOLATION: limit '{k}' must be a positive integer")

        return {
            "max_nodes": raw_limits.get("max_nodes", 64),
            "max_edges": raw_limits.get("max_edges", 256),
            "max_constraints": raw_limits.get("max_constraints", 128),
            "max_operations": raw_limits.get("max_operations", 10000),
            "max_depth": raw_limits.get("max_depth", 64),
        }

    def _parse_graph(
        self, graph: Any, max_nodes: int, max_edges: int
    ) -> tuple[list[str], list[list[str]]]:
        if not isinstance(graph, dict):
            raise ValueError("SCHEMA_VIOLATION: graph must be an object")

        for unsupp in ("weights", "weighted", "hypergraph", "multigraph", "undirected"):
            if unsupp in graph:
                raise ValueError(
                    f"UNSUPPORTED_THEORY: graph property '{unsupp}' outside supported DAG fragment"
                )

        nodes: list[str] = []
        raw_nodes = graph.get("nodes")
        if raw_nodes is not None:
            if not isinstance(raw_nodes, list):
                raise ValueError("SCHEMA_VIOLATION: graph.nodes must be a list")
            for n in raw_nodes:
                if not isinstance(n, str) or not n.strip():
                    raise ValueError("SCHEMA_VIOLATION: graph node must be a non-empty string")
            nodes = list(dict.fromkeys(raw_nodes))

        edges: list[list[str]] = []
        raw_edges = graph.get("edges")
        if raw_edges is not None:
            if not isinstance(raw_edges, list):
                raise ValueError("SCHEMA_VIOLATION: graph.edges must be a list")
            for e in raw_edges:
                if not isinstance(e, (list, tuple)) or len(e) != 2:
                    raise ValueError("SCHEMA_VIOLATION: each edge must be a [from, to] pair")
                if (
                    not isinstance(e[0], str)
                    or not isinstance(e[1], str)
                    or not e[0].strip()
                    or not e[1].strip()
                ):
                    raise ValueError("SCHEMA_VIOLATION: edge endpoints must be non-empty strings")
                edges.append([e[0].strip(), e[1].strip()])

        if not nodes and edges:
            seen: set[str] = set()
            for u, v in edges:
                seen.add(u)
                seen.add(v)
            nodes = sorted(seen)
        elif nodes and edges:
            node_set = set(nodes)
            for u, v in edges:
                if u not in node_set:
                    raise ValueError(
                        f"SCHEMA_VIOLATION: edge references node '{u}' not in graph.nodes"
                    )
                if v not in node_set:
                    raise ValueError(
                        f"SCHEMA_VIOLATION: edge references node '{v}' not in graph.nodes"
                    )

        if len(nodes) > max_nodes:
            raise ValueError(
                f"SCHEMA_VIOLATION: node count {len(nodes)} exceeds max_nodes limit {max_nodes}"
            )
        if len(edges) > max_edges:
            raise ValueError(
                f"SCHEMA_VIOLATION: edge count {len(edges)} exceeds max_edges limit {max_edges}"
            )

        return nodes, edges

    def _parse_order(self, order: Any) -> list[str]:
        if not isinstance(order, list):
            raise ValueError("SCHEMA_VIOLATION: order must be a list of strings")
        result: list[str] = []
        for item in order:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("SCHEMA_VIOLATION: order elements must be non-empty strings")
            result.append(item.strip())
        return result

    def _parse_assignment(self, assignment: Any) -> dict[str, bool]:
        if not isinstance(assignment, dict):
            raise ValueError("SCHEMA_VIOLATION: assignment must be an object")
        result: dict[str, bool] = {}
        for k, v in assignment.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("SCHEMA_VIOLATION: assignment keys must be non-empty strings")
            if not isinstance(v, bool):
                raise ValueError(f"SCHEMA_VIOLATION: assignment value for '{k}' must be a boolean")
            result[k.strip()] = v
        return result

    def _parse_constraints(
        self,
        raw_constraints: Any,
        max_constraints: int,
        record_op: Callable[..., None],
        max_depth: int,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        if not isinstance(raw_constraints, list):
            raise ValueError("SCHEMA_VIOLATION: constraints must be a list")
        if len(raw_constraints) > max_constraints:
            raise ValueError(
                f"SCHEMA_VIOLATION: constraint count {len(raw_constraints)} exceeds limit {max_constraints}"
            )

        parsed: list[dict[str, Any]] = []
        id_map: dict[str, dict[str, Any]] = {}
        all_atoms: set[str] = set()

        for idx, item in enumerate(raw_constraints):
            record_op()
            if not isinstance(item, dict):
                raise ValueError(
                    f"SCHEMA_VIOLATION: constraint item at index {idx} must be an object"
                )

            c_id = item.get("id")
            if c_id is not None:
                if not isinstance(c_id, str) or not c_id.strip():
                    raise ValueError(
                        f"SCHEMA_VIOLATION: constraint id at index {idx} must be a non-empty string"
                    )
                c_id = c_id.strip()
                if c_id in id_map:
                    raise ValueError(f"SCHEMA_VIOLATION: duplicate constraint id '{c_id}'")

            op = item.get("op")
            if op is None or not isinstance(op, str):
                raise ValueError(
                    f"SCHEMA_VIOLATION: constraint item at index {idx} missing valid 'op'"
                )
            op_lower = op.strip().lower()

            if op_lower not in SUPPORTED_OPS:
                raise ValueError(
                    f"UNSUPPORTED_THEORY: operator '{op}' is outside supported DAG+boolean-tree fragment"
                )

            node: dict[str, Any] = {"id": c_id, "op": op_lower}

            if op_lower == "atom":
                atom = item.get("atom")
                if atom is None or not isinstance(atom, str):
                    raise ValueError(
                        f"SYNTAX_INVALID: atom constraint at index {idx} missing 'atom' string"
                    )
                atom_clean = atom.strip()
                if not atom_clean or not ATOM_IDENTIFIER_RE.match(atom_clean):
                    raise ValueError(f"SYNTAX_INVALID: invalid atom identifier syntax '{atom}'")
                if item.get("args"):
                    raise ValueError(
                        f"SYNTAX_INVALID: atom constraint at index {idx} cannot have args"
                    )
                node["atom"] = atom_clean
                all_atoms.add(atom_clean)

            elif op_lower == "not":
                raw_args = item.get("args")
                atom = item.get("atom")
                if raw_args is not None:
                    if not isinstance(raw_args, list):
                        raise ValueError(
                            f"SYNTAX_INVALID: 'not' operator args at index {idx} must be a list"
                        )
                    if len(raw_args) != 1:
                        raise ValueError(
                            f"SYNTAX_INVALID: 'not' operator at index {idx} must have exactly one argument"
                        )
                    node["args"] = raw_args
                elif atom is not None:
                    if not isinstance(atom, str) or not atom.strip():
                        raise ValueError(
                            f"SYNTAX_INVALID: 'not' operator atom at index {idx} must be a string"
                        )
                    atom_clean = atom.strip()
                    if not ATOM_IDENTIFIER_RE.match(atom_clean):
                        raise ValueError(f"SYNTAX_INVALID: invalid atom identifier syntax '{atom}'")
                    node["args"] = [{"op": "atom", "atom": atom_clean}]
                    all_atoms.add(atom_clean)
                else:
                    raise ValueError(
                        f"SYNTAX_INVALID: 'not' operator at index {idx} requires args or atom"
                    )

            elif op_lower in ("and", "or"):
                raw_args = item.get("args")
                if not isinstance(raw_args, list):
                    raise ValueError(
                        f"SYNTAX_INVALID: '{op_lower}' operator at index {idx} requires 'args' list"
                    )
                if len(raw_args) == 0:
                    raise ValueError(
                        f"SYNTAX_INVALID: '{op_lower}' operator at index {idx} requires non-empty args"
                    )
                node["args"] = raw_args

            parsed.append(node)
            if c_id:
                id_map[c_id] = node

        # Resolve ID references and validate no reference cycles
        resolved_nodes: list[dict[str, Any]] = []
        for node in parsed:
            resolved = self._resolve_node_refs(
                node, id_map, set(), all_atoms, record_op, max_depth, depth=0
            )
            resolved_nodes.append(resolved)

        return resolved_nodes, all_atoms

    def _resolve_node_refs(
        self,
        node: dict[str, Any],
        id_map: dict[str, dict[str, Any]],
        visiting: set[str],
        all_atoms: set[str],
        record_op: Callable[..., None],
        max_depth: int,
        depth: int = 0,
    ) -> dict[str, Any]:
        if depth > max_depth:
            raise ValueError(f"RESOURCE_LIMIT: maximum recursion depth ({max_depth}) exceeded")
        record_op()

        node_id = node.get("id")
        if node_id:
            if node_id in visiting:
                raise ValueError(
                    f"SYNTAX_INVALID: cyclic constraint definition detected involving '{node_id}'"
                )
            visiting.add(node_id)

        op = node["op"]
        new_node: dict[str, Any] = {"id": node_id, "op": op}

        if op == "atom":
            new_node["atom"] = node["atom"]
            all_atoms.add(node["atom"])
        elif op == "not":
            arg = node["args"][0]
            if isinstance(arg, str):
                if arg not in id_map:
                    raise ValueError(f"SYNTAX_INVALID: constraint references undefined id '{arg}'")
                child = self._resolve_node_refs(
                    id_map[arg], id_map, set(visiting), all_atoms, record_op, max_depth, depth + 1
                )
            elif isinstance(arg, dict):
                child = self._resolve_inline_node(
                    arg, id_map, set(visiting), all_atoms, record_op, max_depth, depth + 1
                )
            else:
                raise ValueError("SYNTAX_INVALID: invalid argument type for 'not'")
            new_node["args"] = [child]
        elif op in ("and", "or"):
            children: list[dict[str, Any]] = []
            for arg in node["args"]:
                if isinstance(arg, str):
                    if arg not in id_map:
                        raise ValueError(
                            f"SYNTAX_INVALID: constraint references undefined id '{arg}'"
                        )
                    child = self._resolve_node_refs(
                        id_map[arg],
                        id_map,
                        set(visiting),
                        all_atoms,
                        record_op,
                        max_depth,
                        depth + 1,
                    )
                elif isinstance(arg, dict):
                    child = self._resolve_inline_node(
                        arg, id_map, set(visiting), all_atoms, record_op, max_depth, depth + 1
                    )
                else:
                    raise ValueError(f"SYNTAX_INVALID: invalid argument type for '{op}'")
                children.append(child)
            new_node["args"] = children

        return new_node

    def _resolve_inline_node(
        self,
        node: dict[str, Any],
        id_map: dict[str, dict[str, Any]],
        visiting: set[str],
        all_atoms: set[str],
        record_op: Callable[..., None],
        max_depth: int,
        depth: int = 0,
    ) -> dict[str, Any]:
        if depth > max_depth:
            raise ValueError(f"RESOURCE_LIMIT: maximum recursion depth ({max_depth}) exceeded")
        record_op()

        op = node.get("op")
        if not op or not isinstance(op, str):
            raise ValueError("SCHEMA_VIOLATION: inline constraint missing valid 'op'")
        op_lower = op.strip().lower()
        if op_lower not in SUPPORTED_OPS:
            raise ValueError(
                f"UNSUPPORTED_THEORY: operator '{op}' is outside supported DAG+boolean-tree fragment"
            )

        new_node: dict[str, Any] = {"id": node.get("id"), "op": op_lower}
        if op_lower == "atom":
            atom = node.get("atom")
            if not atom or not isinstance(atom, str):
                raise ValueError("SYNTAX_INVALID: inline atom missing 'atom' string")
            atom_clean = atom.strip()
            if not ATOM_IDENTIFIER_RE.match(atom_clean):
                raise ValueError(f"SYNTAX_INVALID: invalid atom identifier syntax '{atom}'")
            new_node["atom"] = atom_clean
            all_atoms.add(atom_clean)
        elif op_lower == "not":
            raw_args = node.get("args")
            atom = node.get("atom")
            if raw_args is not None:
                if not isinstance(raw_args, list) or len(raw_args) != 1:
                    raise ValueError("SYNTAX_INVALID: inline 'not' must have exactly one argument")
                arg = raw_args[0]
                if isinstance(arg, str):
                    if arg not in id_map:
                        raise ValueError(
                            f"SYNTAX_INVALID: constraint references undefined id '{arg}'"
                        )
                    child = self._resolve_node_refs(
                        id_map[arg],
                        id_map,
                        set(visiting),
                        all_atoms,
                        record_op,
                        max_depth,
                        depth + 1,
                    )
                elif isinstance(arg, dict):
                    child = self._resolve_inline_node(
                        arg, id_map, set(visiting), all_atoms, record_op, max_depth, depth + 1
                    )
                else:
                    raise ValueError("SYNTAX_INVALID: invalid argument type for 'not'")
                new_node["args"] = [child]
            elif atom is not None:
                if not isinstance(atom, str) or not atom.strip():
                    raise ValueError("SYNTAX_INVALID: inline 'not' atom must be a string")
                atom_clean = atom.strip()
                if not ATOM_IDENTIFIER_RE.match(atom_clean):
                    raise ValueError(f"SYNTAX_INVALID: invalid atom identifier syntax '{atom}'")
                new_node["args"] = [{"op": "atom", "atom": atom_clean}]
                all_atoms.add(atom_clean)
            else:
                raise ValueError("SYNTAX_INVALID: inline 'not' requires args or atom")
        elif op_lower in ("and", "or"):
            raw_args = node.get("args")
            if not isinstance(raw_args, list) or len(raw_args) == 0:
                raise ValueError(f"SYNTAX_INVALID: inline '{op_lower}' requires non-empty args")
            children: list[dict[str, Any]] = []
            for arg in raw_args:
                if isinstance(arg, str):
                    if arg not in id_map:
                        raise ValueError(
                            f"SYNTAX_INVALID: constraint references undefined id '{arg}'"
                        )
                    child = self._resolve_node_refs(
                        id_map[arg],
                        id_map,
                        set(visiting),
                        all_atoms,
                        record_op,
                        max_depth,
                        depth + 1,
                    )
                elif isinstance(arg, dict):
                    child = self._resolve_inline_node(
                        arg, id_map, set(visiting), all_atoms, record_op, max_depth, depth + 1
                    )
                else:
                    raise ValueError(f"SYNTAX_INVALID: invalid argument type for '{op_lower}'")
                children.append(child)
            new_node["args"] = children

        return new_node

    def _find_roots(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Referenced IDs
        referenced_ids: set[str] = set()

        def scan_refs(n: dict[str, Any]) -> None:
            for child in n.get("args", []):
                if isinstance(child, dict):
                    child_id = child.get("id")
                    if child_id:
                        referenced_ids.add(child_id)
                    scan_refs(child)

        for n in nodes:
            scan_refs(n)

        roots = [n for n in nodes if not n.get("id") or n["id"] not in referenced_ids]
        return roots if roots else nodes

    def _solve_topo(
        self,
        nodes: list[str],
        edges: list[list[str]],
        record_op: Callable[..., None],
    ) -> tuple[bool, list[str] | None, list[str]]:
        node_set = sorted(set(nodes))
        in_degree: dict[str, int] = {n: 0 for n in node_set}
        adj: dict[str, list[str]] = {n: [] for n in node_set}

        for u, v in edges:
            record_op()
            if u == v:
                return False, None, [u]
            adj[u].append(v)
            in_degree[v] += 1

        ready_queue: list[str] = [n for n in node_set if in_degree[n] == 0]
        heapq.heapify(ready_queue)

        order: list[str] = []
        while ready_queue:
            record_op()
            curr = heapq.heappop(ready_queue)
            order.append(curr)
            for neighbor in sorted(adj[curr]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(ready_queue, neighbor)

        if len(order) < len(node_set):
            cycle_nodes = sorted(n for n, deg in in_degree.items() if deg > 0)
            return False, None, cycle_nodes

        return True, order, []

    def _verify_order(
        self,
        nodes: list[str],
        edges: list[list[str]],
        check_order: list[str],
        record_op: Callable[..., None],
    ) -> tuple[bool, str]:
        if set(check_order) != set(nodes) or len(check_order) != len(nodes):
            return False, "provided order does not match graph nodes"

        pos = {n: i for i, n in enumerate(check_order)}
        for u, v in edges:
            record_op()
            if pos[u] >= pos[v]:
                return False, f"dependency edge violated: '{u}' must precede '{v}'"

        return True, ""

    def _eval_node(
        self,
        node: dict[str, Any],
        assignment: dict[str, bool],
        record_op: Callable[..., None],
        max_depth: int,
        depth: int = 0,
    ) -> bool:
        if depth > max_depth:
            raise ValueError(f"RESOURCE_LIMIT: maximum recursion depth ({max_depth}) exceeded")
        record_op()

        op = node["op"]
        if op == "atom":
            atom = node["atom"]
            if atom not in assignment:
                raise ValueError(f"SCHEMA_VIOLATION: missing assignment for atom '{atom}'")
            return bool(assignment[atom])
        elif op == "not":
            return not self._eval_node(node["args"][0], assignment, record_op, max_depth, depth + 1)
        elif op == "and":
            return all(
                self._eval_node(child, assignment, record_op, max_depth, depth + 1)
                for child in node["args"]
            )
        elif op == "or":
            return any(
                self._eval_node(child, assignment, record_op, max_depth, depth + 1)
                for child in node["args"]
            )
        else:
            raise ValueError(f"UNSUPPORTED_THEORY: operator '{op}' outside supported fragment")

    def _eval_all_roots(
        self,
        roots: list[dict[str, Any]],
        assignment: dict[str, bool],
        record_op: Callable[..., None],
        max_depth: int,
    ) -> bool:
        return all(self._eval_node(r, assignment, record_op, max_depth, depth=0) for r in roots)

    def _solve_sat(
        self,
        roots: list[dict[str, Any]],
        free_atoms: list[str],
        assignment: dict[str, bool],
        record_op: Callable[..., None],
        max_depth: int,
    ) -> dict[str, bool] | None:
        def search(index: int) -> dict[str, bool] | None:
            record_op()
            if index == len(free_atoms):
                if self._eval_all_roots(roots, assignment, record_op, max_depth):
                    return dict(assignment)
                return None

            atom = free_atoms[index]
            # Try True first, then False for deterministic ordering
            for val in (True, False):
                assignment[atom] = val
                res = search(index + 1)
                if res is not None:
                    return res
            del assignment[atom]
            return None

        return search(0)
