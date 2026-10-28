"""Group Relative Policy Optimization (GRPO) & Verifiable Rollout Batching (`rlvr_forge.grpo`).

Implements the RLVR advantage estimator:
$$A_i = \\frac{r_i - \\mu_r}{\\sigma_r + \\epsilon} - \\beta_{\\text{KL}} \\cdot D_{\\text{KL}}(\\pi_\\theta \\| \\pi_{\\text{ref}})_i$$
where $r_i$ is the deterministic formal verification reward from `verisci`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List

from rlvr_forge.generator import TaskForge, sample_exploratory_submission
from verisci import DomainTask, VerificationResult, verify_submission


@dataclass
class RolloutSample:
    rollout_id: int
    submission: Dict[str, Any]
    verification: VerificationResult
    raw_reward: float
    kl_divergence: float
    advantage: float


@dataclass
class GRPOGroupBatch:
    task: DomainTask
    samples: List[RolloutSample]
    mean_reward: float
    std_reward: float
    verified_rate: float


def compute_grpo_advantages(
    rewards: List[float],
    kl_divergences: List[float] | None = None,
    beta_kl: float = 0.01,
    eps: float = 1e-6,
) -> List[float]:
    """Compute group-normalized advantages for a group of $G$ rollouts on the same task."""
    if not rewards:
        return []
    n = len(rewards)
    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = math.sqrt(var_r)
    kls = kl_divergences if kl_divergences is not None else [0.0] * n

    advantages: List[float] = []
    for r, kl in zip(rewards, kls):
        norm_adv = (r - mean_r) / (std_r + eps) if std_r > 0 else 0.0
        advantages.append(round(norm_adv - beta_kl * kl, 6))
    return advantages


def build_rlvr_group_batch(
    seed: int,
    domain: str,
    difficulty: int = 3,
    group_size: int = 4,
    beta_kl: float = 0.01,
    split: str = "train",
) -> GRPOGroupBatch:
    """Synthesize a procedural task, sample `group_size` rollouts, verify each via
    `verisci`, and compute GRPO advantages.
    """
    forge = TaskForge(split=split)
    task = forge.generate(seed=seed, domain=domain, difficulty=difficulty)

    submissions: List[Dict[str, Any]] = []
    verifications: List[VerificationResult] = []
    rewards: List[float] = []
    kls: List[float] = []

    for i in range(group_size):
        sub = sample_exploratory_submission(task, rollout_idx=i)
        ver = verify_submission(task, sub)
        submissions.append(sub)
        verifications.append(ver)
        rewards.append(ver.reward)
        # Simulated policy divergence proportional to exploration perturbation
        kls.append(round(0.02 * i, 6))

    advantages = compute_grpo_advantages(rewards, kls, beta_kl=beta_kl)
    n = max(len(rewards), 1)
    mean_r = sum(rewards) / n
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / n)
    verified_rate = sum(1 for v in verifications if v.verified) / n

    samples = [
        RolloutSample(
            rollout_id=i,
            submission=submissions[i],
            verification=verifications[i],
            raw_reward=rewards[i],
            kl_divergence=kls[i],
            advantage=advantages[i],
        )
        for i in range(group_size)
    ]

    return GRPOGroupBatch(
        task=task,
        samples=samples,
        mean_reward=round(mean_r, 6),
        std_reward=round(std_r, 6),
        verified_rate=round(verified_rate, 4),
    )
