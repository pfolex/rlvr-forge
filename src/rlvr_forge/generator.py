"""Deterministic Procedural Task & Reference Trajectory Generator (`rlvr_forge.generator`).

Synthesizes infinite, contamination-free verifiable reasoning tasks across all 4
`verisci` domains (`logic`, `silicon`, `systems`, `matter`) parameterized by:
- cryptographic `seed: int`
- `domain: str`
- `difficulty: int` (1..10)

Also provides:
- `solve_reference(task)`: Produces a known ground-truth optimal solution + action steps.
- `sample_exploratory_submission(task, seed, noise_level)`: Produces diverse candidate
  rollouts (optimal, partially valid, or corrupted) for GRPO training & MCTS search tests.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from typing import Any, Dict, List, Tuple

# Ensure sibling `verisci` package is importable out-of-the-box in the workspace
_VERISCI_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "verisci", "src")
)
if os.path.isdir(_VERISCI_SRC) and _VERISCI_SRC not in sys.path:
    sys.path.insert(0, _VERISCI_SRC)

from verisci.protocol import ActionStep, DomainTask  # noqa: E402


def _derive_rng(seed: int, domain: str, difficulty: int, split: str = "train") -> random.Random:
    digest = hashlib.sha256(f"{split}:{domain}:{difficulty}:{seed}".encode("utf-8")).digest()
    int_seed = int.from_bytes(digest[:8], "big")
    return random.Random(int_seed)


class TaskForge:
    """Procedural, seed-locked synthetic task generator across all 4 `verisci` domains."""

    DOMAINS = ("logic", "silicon", "systems", "matter")

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def generate(self, seed: int, domain: str, difficulty: int = 3) -> DomainTask:
        difficulty = max(1, min(10, int(difficulty)))
        rng = _derive_rng(seed, domain, difficulty, self.split)
        task_id = f"{self.split}-{domain}-d{difficulty}-s{seed}"

        if domain == "logic":
            return self._gen_logic(task_id, difficulty, rng)
        if domain == "silicon":
            return self._gen_silicon(task_id, difficulty, rng)
        if domain == "systems":
            return self._gen_systems(task_id, difficulty, rng)
        if domain == "matter":
            return self._gen_matter(task_id, difficulty, rng)
        raise ValueError(f"Unsupported domain: {domain}")

    def _gen_logic(
        self, task_id: str, difficulty: int, rng: random.Random
    ) -> DomainTask:
        # Chain length scales with difficulty (2..7 inference hops)
        chain_len = min(2 + (difficulty // 2), 7)
        var_names = [f"P{i}" for i in range(chain_len + 1)]
        # Allow randomly negated literals along the implication chain
        polarities = [rng.choice([True, False]) for _ in var_names]

        def lit(idx: int, pos: bool = True) -> str:
            effective = polarities[idx] if pos else (not polarities[idx])
            return var_names[idx] if effective else f"~{var_names[idx]}"

        premises: List[List[str]] = [[lit(0, True)]]
        for i in range(chain_len):
            premises.append(sorted([lit(i, False), lit(i + 1, True)]))

        goal_clause = [lit(chain_len, True)]
        ref_steps = []
        # Step 0 resolves premise 0 and premise 1 -> [lit(1, True)] at index (chain_len + 1)
        for i in range(chain_len):
            p1 = 0 if i == 0 else (chain_len + i)
            p2 = i + 1
            ref_steps.append(
                {"p1": p1, "p2": p2, "conclusion": [lit(i + 1, True)]}
            )

        ref_assignment = {var_names[i]: polarities[i] for i in range(len(var_names))}

        return DomainTask(
            task_id=task_id,
            domain="logic",
            difficulty=difficulty,
            prompt=(
                f"Derive goal clause {goal_clause} from {len(premises)} CNF premises via sound "
                f"propositional resolution and provide a satisfying truth assignment."
            ),
            spec={
                "variables": var_names,
                "premises": premises,
                "goal_clause": goal_clause,
                "require_sat": True,
            },
            metadata={
                "reference_solution": {
                    "proof_steps": ref_steps,
                    "assignment": ref_assignment,
                }
            },
        )

    def _gen_silicon(
        self, task_id: str, difficulty: int, rng: random.Random
    ) -> DomainTask:
        # Choose circuit family based on difficulty and seed
        variant = (difficulty + rng.randint(0, 2)) % 3
        if variant == 0:
            # 3-input Majority Voter + Parity (Full Adder)
            inputs = ["a", "b", "c"]
            outputs = ["sum", "maj"]
            gates = [
                {"op": "XOR", "ins": ["a", "b"], "out": "w_xor1"},
                {"op": "XOR", "ins": ["w_xor1", "c"], "out": "sum"},
                {"op": "AND", "ins": ["a", "b"], "out": "w_and1"},
                {"op": "AND", "ins": ["w_xor1", "c"], "out": "w_and2"},
                {"op": "OR", "ins": ["w_and1", "w_and2"], "out": "maj"},
            ]
            truth_table = []
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        truth_table.append(
                            {
                                "inputs": {"a": a, "b": b, "c": c},
                                "outputs": {
                                    "sum": a ^ b ^ c,
                                    "maj": (a & b) | (c & (a ^ b)),
                                },
                            }
                        )
            max_gates, max_depth = 7, 4
        elif variant == 1:
            # 2-to-1 Multiplexer + Complimentary Parity
            inputs = ["sel", "d0", "d1"]
            outputs = ["y", "p"]
            gates = [
                {"op": "MUX", "ins": ["sel", "d0", "d1"], "out": "y"},
                {"op": "XOR", "ins": ["d0", "d1"], "out": "w_dxor"},
                {"op": "XOR", "ins": ["w_dxor", "sel"], "out": "p"},
            ]
            truth_table = []
            for sel in (0, 1):
                for d0 in (0, 1):
                    for d1 in (0, 1):
                        truth_table.append(
                            {
                                "inputs": {"sel": sel, "d0": d0, "d1": d1},
                                "outputs": {
                                    "y": d1 if sel == 1 else d0,
                                    "p": sel ^ d0 ^ d1,
                                },
                            }
                        )
            max_gates, max_depth = 6, 4
        else:
            # 2-bit Equality & Greater-Than Slice
            inputs = ["a", "b"]
            outputs = ["eq", "gt"]
            gates = [
                {"op": "XOR", "ins": ["a", "b"], "out": "w_neq"},
                {"op": "NOT", "ins": ["w_neq"], "out": "eq"},
                {"op": "NOT", "ins": ["b"], "out": "w_nb"},
                {"op": "AND", "ins": ["a", "w_nb"], "out": "gt"},
            ]
            truth_table = []
            for a in (0, 1):
                for b in (0, 1):
                    truth_table.append(
                        {
                            "inputs": {"a": a, "b": b},
                            "outputs": {
                                "eq": 1 if a == b else 0,
                                "gt": 1 if a > b else 0,
                            },
                        }
                    )
            max_gates, max_depth = 6, 4

        return DomainTask(
            task_id=task_id,
            domain="silicon",
            difficulty=difficulty,
            prompt=(
                f"Synthesize an acyclic digital logic netlist mapping inputs {inputs} "
                f"to outputs {outputs} within {max_gates} gates and depth {max_depth}."
            ),
            spec={
                "inputs": inputs,
                "outputs": outputs,
                "truth_table": truth_table,
                "max_gates": max_gates,
                "max_depth": max_depth,
            },
            metadata={"reference_solution": {"gates": gates}},
        )

    def _gen_systems(
        self, task_id: str, difficulty: int, rng: random.Random
    ) -> DomainTask:
        num_workers = min(2 + (difficulty // 4), 4)
        rounds = 1 + (difficulty // 5)
        required_commits = num_workers * rounds
        resources = ["lock_primary"] if difficulty < 6 else ["lock_primary", "lock_replica"]

        events: List[Dict[str, Any]] = []
        clock = 1
        msg_counter = 0
        for r in range(rounds):
            for w in range(num_workers):
                for res in resources:
                    events.append(
                        {"op": "acquire", "worker": w, "resource": res, "clock": clock}
                    )
                    clock += 1
                events.append({"op": "critical_commit", "worker": w, "clock": clock})
                clock += 1
                for res in reversed(resources):
                    events.append(
                        {"op": "release", "worker": w, "resource": res, "clock": clock}
                    )
                    clock += 1
                nxt_w = (w + 1) % num_workers
                msg_id = f"hb_{r}_{msg_counter}"
                msg_counter += 1
                events.append(
                    {"op": "send", "worker": w, "msg_id": msg_id, "clock": clock}
                )
                clock += 1
                events.append(
                    {"op": "recv", "worker": nxt_w, "msg_id": msg_id, "clock": clock}
                )
                clock += 1

        return DomainTask(
            task_id=task_id,
            domain="systems",
            difficulty=difficulty,
            prompt=(
                f"Synthesize a deadlock-free, linearizable concurrent schedule for "
                f"{num_workers} workers over resources {resources} achieving "
                f"{required_commits} quorum commits with strict Lamport clocks."
            ),
            spec={
                "num_workers": num_workers,
                "required_commits": required_commits,
                "resources": resources,
                "max_buffer": 3,
            },
            metadata={"reference_solution": {"events": events}},
        )

    def _gen_matter(
        self, task_id: str, difficulty: int, rng: random.Random
    ) -> DomainTask:
        # Library of chemically valid, exact-valence target templates
        templates: List[Dict[str, Any]] = [
            {
                "name": "Formaldehyde (CH2O)",
                "formula": {"C": 1, "H": 2, "O": 1},
                "required_motifs": ["C=O", "C-H"],
                "expected_rings": 0,
                "atoms": ["C", "O", "H", "H"],
                "bonds": [
                    {"u": 0, "v": 1, "order": 2},
                    {"u": 0, "v": 2, "order": 1},
                    {"u": 0, "v": 3, "order": 1},
                ],
            },
            {
                "name": "Hydrogen Cyanide (HCN)",
                "formula": {"C": 1, "H": 1, "N": 1},
                "required_motifs": ["C#N", "C-H"],
                "expected_rings": 0,
                "atoms": ["C", "N", "H"],
                "bonds": [
                    {"u": 0, "v": 1, "order": 3},
                    {"u": 0, "v": 2, "order": 1},
                ],
            },
            {
                "name": "Acetaldehyde (C2H4O)",
                "formula": {"C": 2, "H": 4, "O": 1},
                "required_motifs": ["C=O", "C-C"],
                "expected_rings": 0,
                "atoms": ["C", "C", "O", "H", "H", "H", "H"],
                "bonds": [
                    {"u": 0, "v": 1, "order": 1},
                    {"u": 1, "v": 2, "order": 2},
                    {"u": 0, "v": 3, "order": 1},
                    {"u": 0, "v": 4, "order": 1},
                    {"u": 0, "v": 5, "order": 1},
                    {"u": 1, "v": 6, "order": 1},
                ],
            },
            {
                "name": "Cyclopropane (C3H6)",
                "formula": {"C": 3, "H": 6},
                "required_motifs": ["C-C", "C-H"],
                "expected_rings": 1,
                "atoms": ["C", "C", "C", "H", "H", "H", "H", "H", "H"],
                "bonds": [
                    {"u": 0, "v": 1, "order": 1},
                    {"u": 1, "v": 2, "order": 1},
                    {"u": 2, "v": 0, "order": 1},
                    {"u": 0, "v": 3, "order": 1},
                    {"u": 0, "v": 4, "order": 1},
                    {"u": 1, "v": 5, "order": 1},
                    {"u": 1, "v": 6, "order": 1},
                    {"u": 2, "v": 7, "order": 1},
                    {"u": 2, "v": 8, "order": 1},
                ],
            },
        ]
        chosen = templates[(difficulty + rng.randint(0, len(templates) - 1)) % len(templates)]
        return DomainTask(
            task_id=task_id,
            domain="matter",
            difficulty=difficulty,
            prompt=(
                f"Synthesize a valid connected molecular graph for {chosen['name']} "
                f"matching stoichiometry {chosen['formula']} and motifs {chosen['required_motifs']}."
            ),
            spec={
                "formula": chosen["formula"],
                "required_motifs": chosen["required_motifs"],
                "expected_rings": chosen["expected_rings"],
            },
            metadata={
                "molecule_name": chosen["name"],
                "reference_solution": {
                    "atoms": list(chosen["atoms"]),
                    "bonds": [dict(b) for b in chosen["bonds"]],
                },
            },
        )


def solve_reference(task: DomainTask) -> Tuple[Dict[str, Any], List[ActionStep]]:
    """Return `(submission_dict, step_sequence)` for the ground-truth reference solver."""
    ref = dict(task.metadata.get("reference_solution", {}))
    steps: List[ActionStep] = []

    if task.domain == "logic":
        for st in ref.get("proof_steps", []):
            steps.append(
                ActionStep(
                    "derive_clause",
                    {
                        "p1": st["p1"],
                        "p2": st["p2"],
                        "conclusion": list(st["conclusion"]),
                    },
                    rationale=f"Resolve clause {st['p1']} with clause {st['p2']} -> {st['conclusion']}",
                )
            )
        for var, val in sorted(ref.get("assignment", {}).items()):
            steps.append(
                ActionStep(
                    "assign_var",
                    {"var": var, "val": val},
                    rationale=f"Set {var}={val} to satisfy CNF",
                )
            )

    elif task.domain == "silicon":
        for g in ref.get("gates", []):
            steps.append(
                ActionStep(
                    "add_gate",
                    {"op": g["op"], "ins": list(g["ins"]), "out": g["out"]},
                    rationale=f"Synthesize {g['op']} gate driving {g['out']}",
                )
            )

    elif task.domain == "systems":
        for ev in ref.get("events", []):
            params = {k: v for k, v in ev.items() if k != "op"}
            steps.append(
                ActionStep(
                    ev["op"],
                    params,
                    rationale=f"Worker {ev.get('worker')} executes {ev['op']} at Lamport t={ev.get('clock')}",
                )
            )

    elif task.domain == "matter":
        for el in ref.get("atoms", []):
            steps.append(
                ActionStep("add_atom", {"element": el}, rationale=f"Place {el} atom")
            )
        for b in ref.get("bonds", []):
            steps.append(
                ActionStep(
                    "add_bond",
                    {"u": b["u"], "v": b["v"], "order": b["order"]},
                    rationale=f"Bond atom {b['u']} to {b['v']} (order {b['order']})",
                )
            )

    return ref, steps


def sample_exploratory_submission(
    task: DomainTask, rollout_idx: int, noise_prob: float = 0.35
) -> Dict[str, Any]:
    """Generate a candidate rollout for `task` with controlled exploration noise.

    Rollout 0 is guaranteed to be the exact reference solution; higher rollout
    indices introduce realistic structural mutations so GRPO batches exhibit
    non-zero reward variance across group trajectories.
    """
    ref, _ = solve_reference(task)
    if rollout_idx == 0 or noise_prob <= 0.0:
        return ref

    rng = random.Random(hash((task.task_id, rollout_idx)))

    if task.domain == "logic":
        steps = [dict(s) for s in ref["proof_steps"]]
        assign = dict(ref["assignment"])
        if rollout_idx % 3 == 1 and steps:
            # Truncate final step (partial reward)
            steps = steps[:-1]
        elif rollout_idx % 3 == 2:
            # Flip one or more truth assignments
            for k in list(assign.keys())[: max(1, rollout_idx % len(assign))]:
                assign[k] = not assign[k]
        return {"proof_steps": steps, "assignment": assign}

    if task.domain == "silicon":
        gates = [dict(g) for g in ref["gates"]]
        if gates and rollout_idx % 2 == 1:
            idx = rng.randrange(len(gates))
            gates[idx] = dict(gates[idx])
            gates[idx]["op"] = "OR" if gates[idx]["op"] != "OR" else "AND"
        elif gates and rollout_idx % 2 == 0:
            gates = gates[:-1]
        return {"gates": gates}

    if task.domain == "systems":
        events = [dict(e) for e in ref["events"]]
        if events and rollout_idx % 2 == 1:
            # Introduce clock inversion or drop final release
            events = events[:-1]
        elif events:
            events[0] = dict(events[0])
            events[0]["clock"] = 999
        return {"events": events}

    if task.domain == "matter":
        atoms = list(ref["atoms"])
        bonds = [dict(b) for b in ref["bonds"]]
        if bonds and rollout_idx % 2 == 1:
            bonds = bonds[:-1]
        elif bonds:
            bonds[0] = dict(bonds[0])
            bonds[0]["order"] = min(3, bonds[0]["order"] + 1)
        return {"atoms": atoms, "bonds": bonds}

    return ref
