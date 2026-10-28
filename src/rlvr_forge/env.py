"""Gymnasium-style Verifiable Reasoning Environment (`rlvr_forge.env`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rlvr_forge.generator import TaskForge
from verisci import ActionStep, DomainTask, VerificationResult, get_cartridge


@dataclass
class StepOutcome:
    state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class RLVREnv:
    """Multi-step verifiable reasoning environment backed by `verisci` cartridges."""

    def __init__(self, split: str = "train") -> None:
        self.forge = TaskForge(split=split)
        self.task: DomainTask | None = None
        self.state: Dict[str, Any] = {}
        self.steps_taken: List[ActionStep] = []

    def reset(
        self, seed: int = 42, domain: str = "logic", difficulty: int = 3
    ) -> DomainTask:
        self.task = self.forge.generate(seed=seed, domain=domain, difficulty=difficulty)
        self.steps_taken = []
        if domain == "logic":
            self.state = {
                "known_clauses": [list(c) for c in self.task.spec.get("premises", [])],
                "assignment": {},
            }
        elif domain == "silicon":
            self.state = {"gates": []}
        elif domain == "systems":
            self.state = {"events": []}
        elif domain == "matter":
            self.state = {"atoms": [], "bonds": []}
        else:
            self.state = {}
        return self.task

    def step(self, action: ActionStep) -> StepOutcome:
        if self.task is None:
            raise RuntimeError("Call reset() before step().")
        cartridge = get_cartridge(self.task.domain)
        next_state = cartridge.evaluate_step(self.task, self.state, action)
        valid_step = bool(next_state.get("valid_step", False))
        self.state = next_state
        self.steps_taken.append(action)

        shaped_reward = 0.05 if valid_step else -0.05
        return StepOutcome(
            state=dict(self.state),
            reward=shaped_reward,
            done=False,
            info={"valid_step": valid_step, "reason": next_state.get("reason", "")},
        )

    def current_submission(self) -> Dict[str, Any]:
        if self.task is None:
            return {}
        if self.task.domain == "logic":
            proof_steps = [
                s.params for s in self.steps_taken if s.action_type == "derive_clause"
            ]
            return {
                "proof_steps": proof_steps,
                "assignment": dict(self.state.get("assignment", {})),
            }
        if self.task.domain == "silicon":
            return {"gates": list(self.state.get("gates", []))}
        if self.task.domain == "systems":
            return {"events": list(self.state.get("events", []))}
        if self.task.domain == "matter":
            return {
                "atoms": list(self.state.get("atoms", [])),
                "bonds": list(self.state.get("bonds", [])),
            }
        return dict(self.state)

    def verify(self, submission: Dict[str, Any] | None = None) -> VerificationResult:
        if self.task is None:
            raise RuntimeError("Call reset() before verify().")
        sub = submission if submission is not None else self.current_submission()
        cartridge = get_cartridge(self.task.domain)
        return cartridge.verify(self.task, sub)
