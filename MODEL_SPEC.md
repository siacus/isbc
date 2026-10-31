# Consolidated sparse binary factor model

This file is the authoritative specification of the model implemented by
`binary_factors_train.py` with its default `sparse_ae` objective. It records the
architecture and the loss terms that are currently part of the model.

## Architecture and numerical precision

For each document, the selected frozen Hugging Face backbone produces raw token
states. The model-specific pooling rule converts them into one embedding
`x ∈ R^D`; `D` is resolved dynamically and written to the run metadata. The embedding
is never L2-normalized.

The default backbone precision is BF16. The factor head and linear decoder are always
trained in FP32. The head is

`LayerNorm(D) → Linear(D, 512) → GELU → Linear(512, K) → sigmoid(logits / T)`,

with defaults `K=16`, `T=0.5`, and target activity `A=6`, hence `rho=A/K=0.375`.
The final bias is initialized as `T × logit(rho)`, so the initial probability is
actually `rho` after temperature scaling. A straight-through Bernoulli sample creates
the hard code used by the FP32 linear decoder.

The affine decoder is `x_hat = W_dec z + b_dec`. Its intercept is not left at the
framework default. From the same fixed calibration sample used for reconstruction
scaling, let `mu_x` be the coordinate-wise mean embedding. With the seeded initial
decoder weights held fixed and `E[z]=rho 1`, the adopted initialization is

`b_dec(0) = mu_x - W_dec(0) (rho 1)`.

Consequently, the decoder's expected initial reconstruction equals `mu_x`. This
removes a backbone-specific location offset from the optimization problem without
centering, rescaling, or L2-normalizing the raw embeddings. The reconstruction target
remains the original raw pooled embedding. The initialization rule and calibration
statistics are stored in `reconstruction_calibration.json` and checkpoint metadata.

Because the backbone is frozen and held in evaluation mode, its pooled output is a
deterministic input to the SAE. The default trainer computes every document embedding
once, saves the unnormalised FP32 pooled matrix with ordered IDs and provenance, and
reuses that memory-mapped matrix in every epoch and in the final encoding pass. This is
a computational cache, not a change to the architecture or objective: regression tests
against the legacy live-forward path give identical losses and checkpoint tensors for
the same seed. The cache also makes the embedding source and its dynamic dimension an
explicit run artifact.

## Adopted objective

For a minibatch of soft probabilities `p`, straight-through codes `z`, raw pooled
embeddings `x`, and reconstructions `x_hat`, the optimized loss is

`L = 1.0 L_recon + lambda_s(t) L_sparse + 1.0 L_balance + 0.05 L_decor + 0.1 L_entropy`.

The terms are:

- `L_recon = MSE(x_hat, x) / s_x`, where `s_x` is the mean coordinate variance
  estimated once from up to 1,024 training embeddings. This scalar makes the loss
  scale comparable across backbones without altering or normalizing the embeddings.
- `L_sparse = mean_i (mean_k p_ik - rho)^2`. It controls how many bits each document
  uses. Its multiplier starts at `lambda_s(0)=1.0` and adapts during training.
- `L_balance = mean_k (mean_i p_ik - rho)^2`. Its fixed weight is `1.0`; it keeps every
  bit in use and discourages dead or universal bits without allowing this constraint
  to dominate the reconstruction signal.
- `L_decor` is the mean squared off-diagonal covariance of the bit probabilities. Its
  weight is `0.05`, making redundancy reduction an auxiliary pressure rather than the
  principal objective.
- `L_entropy` is the mean conditional Bernoulli entropy
  `-mean[p log p + (1-p) log(1-p)]`. Its weight is `0.1`; minimizing it sharpens soft
  probabilities toward binary decisions so the hard code reflects what was trained.

Every 50 optimizer steps, the document-sparsity multiplier is updated from the
exponential moving average of the relative activity violation:

`v_t = EMA_0.95(|mean_i sum_k p_ik - A| / A)`

`lambda_s(t+1) = clip(lambda_s(t) × exp(0.25 × (v_t - 0.05)), 0.1, 1000)`.

Thus the balance coefficient remains fixed, while only the per-document sparsity
coefficient responds to failure to maintain the six-active-bit target.

## Why these weights are recorded explicitly

The reconstruction scale depends on the selected backbone, and the adaptive sparsity
coefficient changes during optimization. A command line alone is therefore not enough
to reproduce or interpret a run. Every default sparse-AE run records:

- the complete loss specification and fixed weights in `metadata.json` and each
  checkpoint's `run_config.json`;
- the measured reconstruction scale in `reconstruction_calibration.json` and metadata;
- the decoder-intercept initialization rule that mean-matches the initial
  reconstruction to the raw embedding geometry;
- raw and weighted loss terms, the start/end sparsity multiplier, and its final
  violation EMA for every epoch in `train_history.csv`, visualized together in
  `training_loss_paths.png`;
- per-bit decoder-column norms, head output-row norms, biases, monitored-batch
  soft/hard activation rates, and pre-clip gradients every 100 optimizer steps in
  `training_coefficient_metrics.csv`, visualized as
  `bit_coefficient_paths.png`, `bit_activation_paths.png`, and
  `bit_gradient_paths.png`;
- the dynamically resolved embedding dimension and backbone internal width, when
  available.

These records describe the actual objective used by that run even if project defaults
change later.
