//! `rlvr-forge`: Procedural Synthetic Environment & RLVR Advantage Engine.
//!
//! Implements deterministic SplitMix64 seed derivation for zero-contamination
//! procedural task synthesis and Group Relative Policy Optimization (GRPO)
//! advantage normalization with KL divergence regularization.

/// Deterministic SplitMix64 PRNG for contamination-free procedural task synthesis.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Compute Group Relative Policy Optimization (GRPO) normalized advantages:
/// $A_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon} - \beta \cdot \text{KL}_i$
pub fn compute_grpo_advantages(
    rewards: &[f64],
    kl_divergences: &[f64],
    beta_kl: f64,
    eps: f64,
) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    rewards
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let kl = kl_divergences.get(i).copied().unwrap_or(0.0);
            ((r - mean) / (std + eps)) - beta_kl * kl
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_zero_mean() {
        let rewards = vec![1.0, 0.5, 0.2, 0.0];
        let kls = vec![0.0; 4];
        let adv = compute_grpo_advantages(&rewards, &kls, 0.0, 1e-8);
        let sum: f64 = adv.iter().sum();
        assert!(sum.abs() < 1e-6);
        assert!(adv[0] > adv[3]);
    }
}
