# Titans ambiguity register

Paper: Titans: Learning to Memorize at Test Time, arXiv 2501.00663v1.
Baseline code: lucidrains/titans-pytorch v0.6.0, commit 35a549f (tag `baseline-v0.6.0`).
Comparison code: Neuranox/titans-memory v0.3.0, commit 35d4712.

This register lists every choice the paper leaves open or underspecified, in four parts (A–D). Within each table, the rows that move results most come first.

## How we use this file

- We fill in the "Our decision" column as we decide each row, with the date and our initials, e.g. `Per token (BA, 2026-11-10)`.
- We change this file only through a pull request, so every decision has a review and a record.
- When a decision changes code, we put the row ID (e.g. A1) in the commit message or PR title.
- In the report, we cite the row ID wherever we explain a design choice.
- If we find a new ambiguity, we add a row at the end of the right table with the next ID.

## A: Memory update rule

| # | Decision | Paper says | lucidrains / Neuranox | Suggested default | Our decision |
| --- | --- | --- | --- | --- | --- |
| A1 | Chunk size b: the weights each gradient is taken at | §3.2 takes every gradient in a chunk at the chunk-start weights ("inter-chunk non-linear, intra-chunk linear recurrence"); b not given | Refreshed every `batch_size` tokens, default never / every chunk, last layer only | b = 16–64; ablate b | |
| A2 | Granularity of θ, η, α | Token-dependent in the experiments; chunk-level left to future work | θ per token, η and α per chunk / per token, averaged to one scalar | Per token; if memory forces chunks of 4–8, report it | |
| A3 | Which memory weights are fast weights | The whole memory M is updated | All / last layer only | All | |
| A4 | Memory retrieval timing (eq. 15): does token t read `M_{t-1}`, `M_t`, or the last chunk boundary? | Not specified | Last completed chunk (causal) / after the current chunk (leaks) | Last chunk boundary; must pass the causality test | |
| A5 | Memory MLP shape: depth, width, activation, biases | `L_M ≥ 1`, ablated 1–4; main-run depth, width and activation not given | Depth 2, 4× width, GELU, no bias / depth 2, 4× width, SiLU, no bias | `L_M = 2`, 4× width, SiLU; ablate depth | |
| A6 | Is the initial memory `M_0` learned? | Outer loop trains "other parameters"; `M_0` not discussed | Learned per head / last layer detached | Learned | |
| A7 | Gate form and range for θ, η, α | "Data-dependent" functions of `x_t`; no formula, range, bias init or maximum learning rate | sigmoid(linear) per head, θ max 1 (0.1 in `train_mac.py`) / sigmoid, θ ≤ 1 | sigmoid(linear(`x_t`)) per head; sweep θ max over 0.01–1 | |
| A8 | Normalization or residual inside the memory MLP | Not described | LayerNorm plus residual, on by default / none | Off; test on as an ablation | |
| A9 | Loss scaling | `‖M(k) − v‖²₂`, a sum | Mean over features / sum (gradient without the factor 2) | Sum, or mean with θ rescaled; record which | |
| A10 | Backprop through the inner loop, and truncation | Implied (W_K, W_V and gates are learned in the outer loop); truncation not stated | Full backprop within a sequence / through the last layer only | Full backprop within each 4K sequence; detach between sequences | |
| A11 | Memory reset points | Not stated; packed 4K sequences contain document boundaries | Reset per sequence / reset per call | Reset per sequence; optionally at end-of-document tokens | |
| A12 | Multi-head memory | Not mentioned | Supported, default 1 head / single head | 1 head; multi-head as a variant | |
| A13 | Clipping or normalizing the surprise | Not mentioned | Optional soft clamp and spectral norm (on in `train_mac.py`) / none | Off | |

### Open choices for the reference implementation

The Phase 2 reference accepts projected queries, keys, values, and gate values for one stream. These are test configurations. The team can record its baseline choices in the table above after review.

| Row | What the reference makes explicit | What the team needs to settle |
| --- | --- | --- |
| A1 | `gradient_refresh_interval` counts tokens between snapshots of the weights used to evaluate gradients. The fast weights used for retrieval can change inside that interval. | Choose the baseline refresh interval (`batch_size` in `NeuralMemory`). |
| A2 | `update_chunk_size=1` applies momentum and forgetting per token. Larger chunks sum the per-token, learning-rate-weighted gradients at the snapshot, then apply momentum and forgetting once. Reads during an incomplete chunk use the most recent completed state. | Choose per-token or grouped updates for the baseline. Record both intervals and how grouped gate values are obtained. |
| A4 | `before_write` reads the state before consuming the query token. `after_write` includes a write completed by that token. At the last token of a chunk these can return different values even when neither reads a future token. | Choose the boundary read for the baseline. Keep this choice separate from causal attention masking in MAC. |
| A9 | The reference sums squared error over features (eq. 12). Switching to mean error scales the gradient by the inverse of the output dimension. | Choose sum or mean and the corresponding learning-rate scale before comparing results across configurations. |

The reference uses the caller's initial model weights (A6) and already-transformed gates (A7). Upstream comparisons disable the residual/norm wrapper (A8) and surprise clipping (A13) to isolate the update equations. These settings do not settle the architecture or initialization for training.

## B: q, k, v, output and persistent memory

§4.4 names these components but not their exact form, and §3.3 defines persistent memory only as tokens prepended to the input (eq. 19).

| # | Decision | Paper says | lucidrains / Neuranox | Suggested default | Our decision |
| --- | --- | --- | --- | --- | --- |
| B1 | Depthwise-separable conv after the q, k, v projections: kernel size, causality, pointwise part | Present; kernel size not given; Table 5 shows removing it costs about 1.7 perplexity | Absent / kernel 3, non-causal, depthwise only | Causal depthwise conv, kernel 4 | |
| B2 | Order of projection, conv, SiLU and ℓ₂ norm | SiLU on q, k, v and ℓ₂-normalized q, k are stated; order is not | No SiLU, optional RMSNorm / conv then ℓ₂, no SiLU | Projection, conv, SiLU, then ℓ₂ on q and k | |
| B3 | Output normalization and gate before the final projection | Stated (after Mehta et al. 2023); norm type, gate activation and gate input not given | Optional RMSNorm, per-head gate only when heads > 1 / LayerNorm, sigmoid(linear(x)) | RMSNorm, then the output times an activation of linear(x); record the activation | |
| B4 | Persistent memory form | Tokens prepended to the input (eq. 19) | Learned key-value pairs per layer / tokens, prepended and stripped in every block | Tokens, per eq. 19 | |
| B5 | Where persistent tokens are prepended | Shown in each variant's input; per-layer behaviour not stated | Every layer (as keys and values) / every block | Every block; record it | |
| B6 | Does the memory store the persistent tokens? | MAG (eq. 28) and MAL (eq. 30) feed x̃, which includes P, to the memory | No / yes | Follow the equations (yes); compare with no | |
| B7 | Number of persistent tokens `N_p` | Not given | 4 in `train_mac.py` / 16 (presets 8–32) | 4–16; ablate | |

## C: MAC, MAG, MAL and LMM

MAC carries the most open questions, and two of them (C1, C2) decide whether a literal reading of the equations is even causal.

| # | Decision | Paper says | lucidrains / Neuranox | Suggested default | Our decision |
| --- | --- | --- | --- | --- | --- |
| C1 | MAC: causality of retrieved tokens `h_t` | `h_t` is read with the current segment's queries (eq. 21) and placed before the segment (eq. 22) under causal attention, so token i could see `h_j` for j > i; not addressed | No `h_t` / leaks | Read `h_t` with the previous segment's state and queries from the previous segment or learned queries, or mask `h_j` to j ≤ i | |
| C2 | MAC: causality of the output read (eq. 25) | `o_t = y_t ⊗ M*_t(y_t)` reads memory already updated with all of segment t; not addressed | Absent / convex mix with a static memory | Read the state at the last chunk boundary inside the segment, or `M_{t-1}` | |
| C3 | MAC: number of retrieved tokens | Eq. 21 gives one per segment token; Figure 3a's caption shows `N_l` long-term tokens | `N_l` learned tokens / one per token | One per token (eq. 21), or `N_l` pooled queries as a variant | |
| C4 | MAC: what the memory stores (eq. 24) | `y_t`, the attention output; which positions of [P ‖ `h_t` ‖ S] not stated | Pre-attention residual stream / segment positions only (then discarded) | Segment positions of `y_t` only | |
| C5 | MAC: block structure | Not stated whether MAC is one module or a stackable layer, or whether memory is per layer or shared | Memory in chosen layers / memory, attention and FFN in every block | One block per layer, each with its own memory; ablate memory in every layer vs a few | |
| C6 | MAC: segment length | Not given; each segment gets full causal attention | `segment_len` (32 in `train_mac.py`) / 64–128 | 256–512; ablate | |
| C7 | MAC: attention mask over [P ‖ `h_t` ‖ S] | Figure 3a: full causal attention per segment; first `N_p` persistent, then `N_l` long-term tokens | P always visible, block-diagonal causal / P visible, h before S | P visible; h per C1; S causal | |
| C8 | MAC: which positions feed the loss | Not stated (Titans Revisited asks: last chunk or all chunks?) | All tokens / all tokens | All positions, as in standard language modelling | |
| C9 | Positional encoding (all variants) | Not stated | RoPE plus axial embedding / learned absolute, max 8,192 | RoPE inside the attention window | |
| C10 | MAG and MAC: the ⊗ gate | "Any non-linear gating"; MAG normalizes y and M(x̃) with learnable vector weights, then applies σ | attention output × sigmoid(memory output) / convex mix | o = Norm₁(y) ⊙ σ(Norm₂(M(x̃))), with Norm₁ and Norm₂ learned-scale RMSNorms; record it | |
| C11 | MAG and MAL: sliding-window size | Not given | `segment_len` plus memory tokens / 512 | Pick two (e.g. 512 and 2,048) and ablate | |
| C12 | MAG vs MAL: persistent-token prefix in sliding attention | MAG uses SW-Attn* (with prefix, eq. 27); MAL's eq. 31 uses SW-Attn without the star | Prefix always visible / prefix visible in both | Prefix visible in both; note it | |
| C13 | MAG: branch layout per block | Two branches, described as a multi-head design with differently structured heads | Gate on the attention output / parallel branches, then FFN | Parallel branches in each block, then a feed-forward layer | |
| C14 | MAL: block layout | "H3-like": the sequence model replaced by the memory; layer ratio and FFN placement not given | Memory residual before attention in chosen layers / memory, then attention, then FFN per block | Memory, then attention, then FFN per block | |
| C15 | LMM-only block design | A standalone sequence model; block not described | n/a / memory plus FFN | Pre-norm memory layer plus a gated feed-forward layer | |

## D: Training and evaluation

The paper states the optimizer recipe and data but not model shapes, and it reuses some baseline numbers from Yang et al. (2024a), the Gated DeltaNet paper, rather than retraining them.

| # | Decision | Paper says | lucidrains / Neuranox | Suggested default | Our decision |
| --- | --- | --- | --- | --- | --- |
| D1 | Model shapes at each size (layers, width, heads) | Sizes 170M, 340M, 400M and 760M; shapes not given; training "follows Yang et al. (2024a)" | Example script only / presets of 12–32 layers, widths 512–1,024 | Copy the Gated DeltaNet configs at matching sizes | |
| D2 | Baselines | Appendix B: 400M baseline results taken from Yang et al. (2024a) | n/a | Train our own baselines in the same codebase | |
| D3 | Unstated optimizer details: warmup, Adam betas, gradient clipping | AdamW, lr 4e-4, cosine schedule, 0.5M-token batches, weight decay 0.1 are stated; the rest is not | AdoptAtan2, lr 2e-4 in `train_mac.py` / AdamW with warmup helper | Stated values plus the Gated DeltaNet recipe's other settings | |
| D4 | Numeric precision of the inner loop | Not stated | Follows input dtype / follows input dtype | fp32 for memory weights and scans, bf16 elsewhere | |
| D5 | Data at our scale | FineWeb-Edu, 15B tokens (30B for 760M), Llama 2 tokenizer, 32K vocab, 4K context; the depth study uses a Pile subset | Enwik8 sample / none | FineWeb-Edu sample, Llama 2 tokenizer, 4K context; scale tokens to our budget | |
| D6 | Evaluation protocol | Wikitext, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ; harness version and acc vs acc_norm not given | None / none | lm-evaluation-harness, pinned version; follow Gated DeltaNet's metric choices | |
| D7 | Needle-in-a-haystack setup | S-NIAH from RULER at 2K–16K with 4K-trained models; model size not given | None / none | RULER defaults; report model size and training length | |
| D8 | BABILong fine-tuning | MAC fine-tuned; size, data and steps not given; baselines from Kuratov et al. (2024) | None / none | Out of scope unless fully documented | |
| D9 | The "larger than 2M context" claim | Stated in the conclusion; setup not given | None / none | Out of scope; test what our positional scheme allows | |
| D10 | Efficiency comparison | Throughput vs sequence length; hardware and kernels not given; MAL uses FlashAttention | Flex attention optional / pure PyTorch | Report hardware; compare only against same-codebase baselines | |

## Sources

- Titans paper: https://arxiv.org/abs/2501.00663
- lucidrains/titans-pytorch: https://github.com/lucidrains/titans-pytorch
- Neuranox/titans-memory: https://github.com/Neuranox/titans-memory
- Titans Revisited (arXiv 2510.09551): https://arxiv.org/abs/2510.09551
