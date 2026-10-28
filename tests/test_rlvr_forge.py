"""Unit test suite for `rlvr-forge` procedural generation, reference solving, env stepping, and GRPO."""

from __future__ import annotations

import unittest

from rlvr_forge import (
    RLVREnv,
    TaskForge,
    build_rlvr_group_batch,
    compute_grpo_advantages,
    solve_reference,
)
from verisci import verify_submission


class TestRLVRForge(unittest.TestCase):
    def test_deterministic_seed_and_train_eval_disjointness(self) -> None:
        train_forge = TaskForge(split="train")
        eval_forge = TaskForge(split="eval_holdout")

        t1 = train_forge.generate(seed=101, domain="logic", difficulty=4)
        t2 = train_forge.generate(seed=101, domain="logic", difficulty=4)
        e1 = eval_forge.generate(seed=101, domain="logic", difficulty=4)

        self.assertEqual(t1.to_dict(), t2.to_dict())
        self.assertNotEqual(t1.task_id, e1.task_id)

    def test_reference_solver_achieves_100_percent_verification_across_all_domains(self) -> None:
        forge = TaskForge(split="train")
        for domain in TaskForge.DOMAINS:
            for difficulty in (1, 3, 6, 9):
                task = forge.generate(seed=77 + difficulty, domain=domain, difficulty=difficulty)
                ref_sub, steps = solve_reference(task)
                res = verify_submission(task, ref_sub)
                self.assertTrue(
                    res.verified,
                    f"Reference solver failed for {domain} d={difficulty}: {res.failed_invariants}",
                )
                self.assertAlmostEqual(res.reward, 1.0)
                self.assertGreater(len(steps), 0)

    def test_gym_environment_step_by_step_replay(self) -> None:
        env = RLVREnv(split="train")
        for domain in TaskForge.DOMAINS:
            task = env.reset(seed=2026, domain=domain, difficulty=3)
            _, ref_steps = solve_reference(task)
            for action in ref_steps:
                outcome = env.step(action)
                self.assertTrue(outcome.info["valid_step"])
            final_ver = env.verify()
            self.assertTrue(final_ver.verified)
            self.assertAlmostEqual(final_ver.reward, 1.0)

    def test_grpo_advantage_normalization_and_batch_builder(self) -> None:
        advs = compute_grpo_advantages([1.0, 0.5, 0.25, 0.0], beta_kl=0.0)
        self.assertAlmostEqual(sum(advs), 0.0, places=5)
        self.assertGreater(advs[0], advs[-1])

        for domain in TaskForge.DOMAINS:
            batch = build_rlvr_group_batch(
                seed=99, domain=domain, difficulty=4, group_size=4
            )
            self.assertEqual(len(batch.samples), 4)
            # Sample 0 is the reference trajectory -> always verified
            self.assertTrue(batch.samples[0].verification.verified)
            # Sample 0 has the highest advantage in the group
            self.assertGreater(batch.samples[0].advantage, batch.samples[-1].advantage)


if __name__ == "__main__":
    unittest.main()
