# Original contribution opportunities for HERMES

Research date 19 November 2026. This is a targeted novelty assessment and pilot proposal. It does not establish comprehensive novelty. The user has one rented GPU for a few days, and the professor requires an original method or improvement. Venue eligibility will depend on a forthcoming approved-venue document.

## What does the existing evidence establish and what does it leave open

### Takeaway

The project has an audited reproduction and control foundation. A publication claiming an original improvement still needs a precise mechanism, causal evidence of its benefit, comparisons to recent methods and independent confirmation. The small full-benchmark difference alone does not establish such a contribution.

### Cited Findings

- The full real-time visual benchmark contains 498 videos and 2,495 questions per method. HERMES 4K scores 1,555 correct, FIFO 4K scores 1,546. The observed +0.3607-point difference has a video-cluster 95% interval of −0.4810 to +1.1628 points. The current study uses LLaVA-OneVision-Qwen2-0.5B and a full 4K comparison. [Local project status](../../PROJECT_STATUS.md)
- The real-video diagnostic uses one 16-frame prefix, 24 layers and 3,136 visual tokens. Current-versus-native cosine is 0.5593 and retained-set Jaccard is 0.6503. Corrected-attention QA effects remain unmeasured. Component ablations are measured on a 50-video shard and all paired intervals include zero. [Local project status](../../PROJECT_STATUS.md)
- The maintained scorer holds pseudo-query input embeddings fixed across layers and omits a causal mask in the manual attention path. This is a specific implementation approximation to investigate, and correcting it alone would not establish a new algorithmic contribution. [Scorer implementation](../../../inference/llavaov_hermes.py)
- The literature directory currently contains 51 PDF files. This inventory includes both named papers and identifiers. The file count is not a count of unique papers fully reviewed. The previous report also contains non-video topics. [Local literature directory](../../related_papers/)
- HERMES introduces a training-free hierarchical interpretation of the KV cache and maintains it before unknown future user queries arrive. [HERMES](https://arxiv.org/abs/2601.14724)

### Inferences

- The 0.5B model is an appropriate cheap mechanism-development platform. A larger or second backbone would strengthen a final paper, but a large grid is unsuitable as the immediate experiment under the stated compute limit.
- Existing aggregate comparisons leave scorer fidelity, backbone capability, sparse sampling and information loss through retention entangled. Native causal attention is a reference for numerical computation, and actual QA improvement must be measured separately.
- Approximately 93% of measured video wall time lies outside the encoding/compression/answer timers according to the parent's raw-event recomputation. This residual needs explicit profiling. A compression-path optimization can still be meaningful, while a large total-wall-speedup claim would require evidence.
- The whole StreamingBench result has already influenced design. A method developed now should use a fixed video-disjoint development subset and retain a separate confirmation set or benchmark. Describing already-inspected results as untouched test evidence would overstate independence.

### Gaps

- No method here has been demonstrated to improve accuracy or to achieve an accuracy–compute Pareto improvement.
- No GPU time budget in exact hours or hardware specification has been committed. A few days does not translate reliably to a fixed number of runs before model loading, decoding and pilot throughput are measured.
- The professor's approved venues and ranking rule remain pending. The original contribution should be specified independently of an assumed acceptance probability.

## Which prior work overlaps the apparent research directions

### Takeaway

Generic adaptive memory, query banks, recency gates, uncertainty-triggered retrieval, temporal coverage and summary prototypes already have substantial prior art. A candidate should be described by its exact mechanism and evaluated against its closest overlap. The existing EDMI proposal has no verified novelty or measured implementation.

### Cited Findings

| Primary source | Verified mechanism relevant to this project | Implication for candidate novelty |
|---|---|---|
| HERMES, January 2026 | Hierarchical bounded KV memory and pre-query management | Baseline being reproduced. A threshold change or scorer correction needs a further contribution. [Paper](https://arxiv.org/abs/2601.14724) |
| InfiniPot-V, 2025 | Temporal redundancy removal plus value-norm token ranking under a length-independent cap | Novelty cannot be based simply on removing similar frames or combining redundancy and salience. [Paper](https://arxiv.org/abs/2506.15745) |
| StreamMem, 2025 | Auxiliary generic-query attention, input frame filtering and frame-wise KV prototypes | Query-agnostic saliency and pre-encoding filtering already exist. Its method section recomputes auxiliary queries with incoming segments. [Full paper](https://arxiv.org/html/2508.15717v1) |
| SAVEMem, May 2026 | Fixed pseudo-question bank, three-tier memory, semantic token scoring and query-conditioned recency gating | A generic query bank plus recent/old memory split is heavily overlapping. Its semantic MaxSim prior is instantiated once and reused, which matters for any claim of reusable scoring. [Full paper](https://arxiv.org/html/2605.07897v1) |
| MuKV, CVPR 2026 | Patch/frame/segment KV compression with attention/frequency signals and hierarchical retrieval | Multi-granularity compression needs more specific differentiation. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_MuKV_Multi-Grained_KV_Cache_Compression_for_Long_Streaming_Video_Question-Answering_CVPR_2026_paper.html) |
| OASIS, CVPR 2026 | Event hierarchy and two-stage reasoning that retrieves older evidence when uncertainty arises | An entropy trigger for old-memory recall alone is insufficiently distinct. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.html) |
| ProtoKV, ICML 2026 | Exact near window and fixed semantic–spatial prototype state with residual statistics for delayed queries | Generic summary-state memory and delayed-evidence motivation already have a close comparator. This paper is in the local collection as minh26a.pdf. [PMLR paper](https://proceedings.mlr.press/v306/minh26a.html) |
| DSCache, May 2026 | Decoupled cumulative-past and on-demand instant KV construction | The distinction between cache construction and eviction already has a direct method. [Paper](https://arxiv.org/abs/2605.01858) |
| CoRDS, May 2026 | Joint key/value coreset coverage and orthogonality-based diversity | Temporal/representation coverage should be compared to this principled selection baseline. [Paper](https://arxiv.org/abs/2605.14310) |
| DeltaS, September 2026 | State drift from gated-delta linear attention as a cheap retention signal | A generic novelty/drift signal is already studied, although its hybrid backbone state differs from the current Qwen2 backbone. [Paper](https://arxiv.org/abs/2609.27470) |
| Video-MemComp, October 2026 | Pre-RoPE aggregation with attention-conserving bias and sensitivity-driven layer budgets | RoPE-safe averaging and adaptive layer allocation already have relevant prior work. [Paper](https://arxiv.org/abs/2610.32784) |
| KV², October 2026 | Cheap proxy selects sparse context queries, then causal selective reconstruction produces eviction scores | Sparse accurate probe computation already exists in reusable text caches. Its limitations explicitly propose adaptive refinement schedules. [Full paper](https://arxiv.org/html/2610.03198v2) |
| LORE-KV, October 2026 | Samples short future continuations and uses projected deletion cost with trajectory weighting | Generic future-query ensembles, reliability weighting and disagreement are already close concepts. Any prior suggestion that these are inherently new must be revised. [Paper](https://arxiv.org/abs/2610.07643) |
| RestoreKV, August 2026 | Self-distillation changes retained representations through small prefill-only adapters | Lightweight learned cache repair also has prior art. [Paper](https://arxiv.org/abs/2608.01247) |

### Inferences

- The most defensible immediate research lead is the specific fidelity–cost gap measured in the present HERMES implementation. Its investigation uses the existing code and can produce a cheap decision about whether a larger method effort is worthwhile.
- Generic semantic memory and entropy-triggered retrieval are crowded directions. EDMI should retain its status as a historical hypothesis until a precise distinction from OASIS, ProtoKV and other recall/summary mechanisms is established.
- Reproduction, a parameter change, a new acronym, or correcting missing causal propagation alone would not provide a convincing original-method claim for an ambitious journal. These can support a stronger new mechanism or rigorous analysis.
- The comparison matrix should include timing placement. A method that changes query-time retrieval has different latency obligations from one that maintains memory before questions arrive.

### Gaps

- This is a targeted primary-source review. Full equations, code and all versions have not been exhaustively reconciled. Several arXiv HTML renderings include inconsistent internal date strings, so the arXiv submission/version metadata should be used for chronology.
- Publication claims require inspecting the nearest methods' implementations and any later versions available at submission time.
- The current broad summary does not prove that the collection covers the most recent related work. The new October references extend the old catalogue.

## Which narrow mechanism is worth a cheap pilot

### Takeaway

Prioritize a causally faithful probe reference followed by selective refresh of cached propagated probe states. Treat this as a provisional research candidate with a strict stop rule. Two fallback directions are listed to make the decision explicit, and their overlap/compute risks are higher.

### Cited Findings

- The current local diagnostic already establishes a selection discrepancy. It does not establish that a faithful scorer improves QA. [Project status](../../PROJECT_STATUS.md)
- KV² uses sparse causal reconstruction, and its adaptive-refinement discussion overlaps any claim based solely on stopping extra scoring when scores stabilize. [Method and limitations](https://arxiv.org/html/2610.03198v2)
- SAVEMem reuses a fixed semantic prior, while StreamMem recomputes auxiliary-query attention during segment processing. A new mechanism must distinguish contextual propagated-probe reuse from a static similarity prior. [SAVEMem](https://arxiv.org/html/2605.07897v1), [StreamMem](https://arxiv.org/html/2508.15717v1)
- ProtoKV and Video-MemComp already address summary representations and aggregation error, limiting a generic new summary-token proposal. [ProtoKV](https://proceedings.mlr.press/v306/minh26a.html), [Video-MemComp](https://arxiv.org/abs/2610.32784)

### Inferences

**Candidate 1, preferred for initial investigation — staleness-controlled refresh of causally propagated generic probes.**

Mechanism proposal. First implement native-equivalent causal probe scoring as an experimental reference. Store the small layer-wise propagated query states from that pass, then reuse them to score a changing visual cache with cheap query/key products. Trigger a fresh causal probe propagation when an online staleness criterion predicts that selection quality has deteriorated. A fixed recent-token floor can serve as a safety policy. The future real question must be unavailable to the trigger.

Potential distinct contribution. A validated online criterion controlling when contextual probe states require refresh as a stream and its compressed cache evolve. This differs in intended setting from a one-time text-cache reconstruction and from a static pseudo-question embedding bank. The contribution would need a clear derivation or calibrated bound, measured cache/set fidelity and a demonstrated QA–compression-cost benefit. Novelty is provisional.

Implementation issues that must be handled. HERMES's pseudo-question text changes after an answered question because it incorporates conversation history. Such a text change must invalidate probe states or select a separately keyed state. RoPE logical positions change with stream growth, and compaction/reindexing can invalidate positional alignment. Eviction changes retained cache entries, so state/score reuse must be keyed by stable token identities or explicitly invalidated. Query projections cached in rotated coordinates need correct position transforms. The cheap trigger's own computation, storage and synchronizations must be included in the cost. Cached contextual states add memory and belong inside the total memory accounting.

Falsifiable hypothesis. Within a fixed cache and scoring-compute budget, adaptive refresh preserves the useful selection/QA behavior of a fresh causal probe reference more effectively than fixed-interval refresh. Compare against original HERMES and FIFO to establish practical relevance. Selection Jaccard alone is an intermediate metric.

Cheap pilot and stop rule. Use CPU/manual inspection of a small set of existing HERMES-only, FIFO-only and shared failures first. Choose 20–30 videos with complete prefixes and a task/temporal mix, with a video-disjoint development/confirmation split selected before new runs. Run a small same-session comparison of original HERMES, FIFO and fresh causal scoring at one tight budget. Measure actual wall time on 2–3 videos before committing the pilot. If fresh causal scoring changes token sets but produces no promising QA/evidence-preservation signal, stop the refresh-method effort. If it is promising, add cached fixed-interval refresh and one predeclared adaptive rule on the same pilot. Count refresh calls, peak memory, compression time, p95 event latency, accuracy and failure types. A few days of one GPU supports a pilot decision, not a promised completed Q1 validation grid.

**Candidate 2, conditional fallback — multiplicity-aware repeated summary consolidation.**

Mechanism proposal. Investigate whether repeatedly folding an existing summary token together with fresh evicted tokens treats the already-aggregated history as one token and progressively changes its effective mass. A representation can carry support mass and temporal support metadata and combine summaries using mass-preserving aggregation. A stronger method would need a specific treatment of temporal event identity or aggregation uncertainty, beyond the algebra of weighted means.

Potential distinct contribution. An associative, bounded-memory consolidation rule whose error under repeated eviction is characterized and whose event/count preservation is validated. Distinguish support-token multiplicity from event count. Ten similar frames do not represent ten distinct events. Compare with HERMES summary folding, a weighted-mean-only control, ProtoKV and Video-MemComp. Their summary-state and attention-conservation mechanisms make overlap risk high.

Cheap pilot and stop rule. Begin with constructed cache/stream tests that vary the number of repeated consolidation steps while holding underlying evidence fixed. Synthetic tests establish behavior of the operator and remain separate from real QA evidence. Manually check whether selected real failures contain a sampled old event that is lost in summary consolidation. Stop if the failure arises before visual sampling or if weighted means fully explain the benefit without a further contribution. Query-time attention-bias changes may require slower backends, so account for kernel compatibility.

**Candidate 3, lower priority — evidence-aware allocation between temporal sampling and retention.**

Mechanism proposal. If shared counting/event failures are traced to sub-two-second evidence missed at 0.5 fps, allocate a fixed encoded-frame budget to brief temporal changes and maintain a bounded cache. Evaluate against uniform and simple change-based sampling at the same number of encoded frames and cache bytes. Use only observed stream information.

Potential distinct contribution. A narrowly defined online controller with an evidence-recovery or temporal-aliasing objective. Input filtering, novelty sampling, multi-granularity representations and event memory are already established. The observed poor counting score alone does not prove an opportunity or novelty. StreamMem, InfiniPot-V, MuKV and temporal-video sampling literature require detailed comparisons.

Cheap pilot and stop rule. Inspect short intervals around 10–15 sampled real failures, then use a small diagnostic with denser sampling to test whether missing temporal observations explain errors. Denser sampling is a diagnostic upper bound with higher cost. The proposed controller must subsequently match the sampling/encoding budget. Stop if the needed information was present but the backbone still fails. This candidate risks expanding the project into two problems and is less compatible with a few-day GPU limit.

**Minimum decision package for the next GPU rental.**

1. Freeze current results and write a one-page pilot protocol before running new inference.
2. Inspect evidence intervals on CPU before choosing a mechanism.
3. Use the existing 0.5B implementation for initial experiments, one budget and a small video sample.
4. Include a higher-budget or native feasible-prefix diagnostic only for selected cases, with an explicitly matched context range.
5. Measure actual throughput early and reserve GPU time for confirming the best candidate rather than enumerating many variants.
6. Scale to another model or benchmark only after a convincing pilot and a resource estimate. These would strengthen a final submission, while the current budget does not support promising them immediately.

### Gaps

- No candidate has been implemented, tested or certified original in this research pass.
- Candidate 1's cheap staleness feature is a research object requiring derivation and calibration. It cannot be assumed to bound answer error.
- Native probe superiority is unmeasured. Its absence would weaken the strongest motivation for candidate 1.
- Candidate 2 could reduce to a standard weighted-aggregation correction, and candidate 3 has extensive adjacent literature. Both require a sharper mechanism before an original-method claim.
- Acceptance depends on actual results, novelty, scope and review. A Q1 label does not supply evidence that any candidate will be accepted.
