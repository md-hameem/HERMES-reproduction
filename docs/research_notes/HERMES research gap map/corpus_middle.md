# HERMES literature gap review — middle corpus

Review date: 2026-11-19. Scope: 17 local PDFs, the case-insensitive filename-sorted slice [17:34] of docs/related_papers. All 17 yielded usable text; 521 total PDF pages are accounted for, **not 521 pages read exhaustively**. Methods and relevant evaluation/limitations were selectively read. Gemini, the agent-memory survey and coding-agent analysis received an explicitly narrower skim. Page references mean one-based local PDF pages. SHA256 and version identity are preserved in the JSON ledger. No numerical results were independently reproduced in this literature pass. Unknown/no limitation located does not mean no limitation exists.

## Which gaps survive this part of the corpus?

### Takeaway

The strongest defensible opportunities are constrained comparisons and a narrowly specified mechanism. Adaptive hierarchy, semantic probe banks, retrieval, scene changes, layer budgets, quantization and efficient context memory are extensively occupied. A paper cannot claim originality merely by combining those names.

### Cited Findings

- ReKV explicitly flags growing external KV storage, fixed blocks and fixed retrieval counts (local p16). StreamKV later introduces semantic segments and layer-adaptive compression; SAVEMem adds query-dependent retrieval scope. Treat the old limitations as partly addressed, not newly discovered.
- InfiniPot-V explicitly leaves fixed TaR/VaN allocation and multimodal compression open (p24), but FluxMem, FreshMem, CurveStream and SAVEMem already address several adaptive-memory variants. InfiniPot-V also measures edge energy (p10), so energy evaluation alone is not a unique contribution.
- SAVEMem already uses a fixed pseudo-question bank to guide bounded, three-tier visual memory and query-time retrieval (pp4–5), and explicitly acknowledges pseudo-streaming that ignores encoding latency (p15). This is particularly close prior art for generic multi-probe retention. [Primary paper](https://arxiv.org/abs/2605.07897).
- StreamingTOM says the visual encoder dominates end-to-end latency despite active-cache improvements (p14). MemStream shows accuracy gains are task dependent: external retrieval hurts VideoMME slightly and RVS-Movie falls despite RVS-Ego gains (p7).
- Current primary-source crosscheck: [LORE-KV](https://arxiv.org/abs/2610.07643) uses sampled frozen-model continuations and response-side query states; [KV²](https://arxiv.org/abs/2610.03198) selectively reconstructs proxy-selected context for reusable eviction scoring; [Video-MemComp](https://arxiv.org/abs/2610.32784) accounts for compression sensitivity and pre-RoPE aggregation. Only their abstracts/metadata were checked online in this shard; full overlap analysis remains necessary before claiming novelty.

### Inferences

1. **Comparative unresolved: evidence survival across age and repeated compression.** Many methods optimize salience/variation or retrieved relevance. A controlled test should ask whether rare static text, identity, temporal order and distant episodic details survive repeated decisions at the same total memory. This is an evaluation gap hypothesis across selected papers, not a proof that nobody has tested it.
2. **Comparative unresolved: active GPU versus total retained state.** Compare RAM/disk growth, embeddings, probe states, quantized stores, retrieval indices and transfer overhead as well as KV. Bounded GPU memory does not imply bounded system memory.
3. **Published limitation / evaluation improvement: live arrival cost.** Queueing, encoder work, dropped/late frames and response lag should count. This can strengthen evidence but needs a method addressing the bottleneck to be an original-method thesis.
4. **Speculative method candidate: native causal scorer reuse with explicit invalidation.** Cache history-conditioned probe states and refresh on measurable staleness after eviction/context changes. Its possible distinction is the online reuse/invalidation rule under causal history and a full auxiliary-state budget. Generic probes, caching, selective refresh or reconstruction are insufficient novelty claims. Compare SAVEMem, FlexMem/MemIndex, LORE-KV and KV² carefully.
5. **Implementation validation before originality.** If the local HERMES scorer differs from the backbone's causal layer propagation, correcting it is a reproduction correction first. The parent's local audit supplies measured discrepancy; this shard does not independently establish an accuracy benefit of fixing it.

### Gaps

- No exhaustive external citation graph or full newest-paper/code audit was completed here. The ledger is a bounded, primary-text review of the assigned 17 files.
- Absence of a named staleness rule in these selected passages does not prove absence from the wider literature.
- A cheap first experiment is a small held-out causal-prefix comparison: current scorer, corrected native reference, cached reference with/without refresh, FIFO and a fixed-semantic-bank control. Report native-score agreement, selected-token agreement, paired QA, evidence age, peak total memory and complete cost. Stop pursuing a complex method if faithful scoring does not yield useful QA gains.
- Split tuning/development from final tests. Bootstrap by video, stratify evidence lag/task, report uncertainty and failures, and maintain identical sampling/position semantics. Scale to one feasible stronger backbone only after a useful reproducible effect.

## What does each assigned PDF establish?

### Takeaway

The following records preserve every assigned file, including supporting and peripheral material. More detailed machine-readable fields are in corpus_middle.json.

### Cited Findings

#### Titans: Learning to Memorize at Test Time

File: `2501.00663v1.pdf`; 27 PDF pages; relevance: **supporting**. [Primary source](https://arxiv.org/abs/2501.00663v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Neural memory trained to store historical context through test-time updates; three integration architectures.

**Mechanism:** Surprise-driven nonlinear associative memory plus short-term attention and persistent memory; MAC/MAG/MAL (Sections 3–4).

**Evaluation:** Language modeling, reasoning, needle retrieval, genomics and time series; not streaming VideoQA.

**Author limitations/observations:** p6: more expressive memory architectures suggested as future work.

**Analyst inferences:** Analyst inference: Training and test-time memory learning make this a different intervention from a frozen VLM cache selector.

**Candidate gaps:** Speculative transfer: learned surprise versus cheap evidence-preserving retention; not a claim that surprise memory is original.

**Already occupied / partly addressed:** Titans itself occupies neural surprise-based memory.

**Cheap test:** Use a frozen surprise proxy on a small held-out video slice; count all extra forward-pass cost.

#### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

File: `2501.05510v2.pdf`; 29 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2501.05510v2).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Timestamped online evaluation across backward tracing, real-time understanding and forward active responding.

**Mechanism:** Benchmark protocols rather than retention policy.

**Evaluation:** Three scenarios / twelve tasks; latency analysis p8.

**Author limitations/observations:** p13 Section 9: limited suitable annotations, automatic QA quality and annotation cost constrain diversity and may bias data. p13: offline models can score through trivial guesses in forward active responding; benchmark fairness limitation.

**Analyst inferences:** Analyst inference: Aggregate accuracy can conceal distant-memory failures and active-response timing costs.

**Candidate gaps:** Comparative unresolved: test retention methods separately on backward tracing, present perception and active response, with lag strata.

**Already occupied / partly addressed:** OVO already supplies this benchmark; a new evaluation slice is not automatically a new method.

**Cheap test:** Run a preregistered small OVO backward-tracing subset with paired predictions and timestamp assertions.

#### Qwen2.5-VL Technical Report

File: `2502.13923v1.pdf`; 23 PDF pages; relevance: **supporting**. [Primary source](https://arxiv.org/abs/2502.13923v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Dynamic-resolution multimodal models with absolute-time-aware video position encoding.

**Mechanism:** Dynamic ViT and temporal alignment of MRoPE to absolute timestamps (p5).

**Evaluation:** 3B/7B/72B broad image, document, grounding and video tasks.

**Author limitations/observations:** No dedicated author limitation passage identified in selected sections; do not interpret architectural discussion of older systems as this model's limitation.

**Analyst inferences:** Analyst inference: Native long video context does not establish constant total memory for unbounded streams. Analyst inference: Changing token selection without preserving time/position semantics can change model behavior.

**Candidate gaps:** Measured-local-validation requirement: preserve timestamp and logical position identity during cache compaction; then evaluate cross-backbone transfer.

**Already occupied / partly addressed:** Absolute temporal alignment is already part of Qwen2.5-VL.

**Cheap test:** Compare uncompressed prefix output with identity-retention path, then selected retention under identical timestamps.

#### Streaming Video Question-Answering with In-context Video KV-Cache Retrieval

File: `2503.00540v1.pdf`; 16 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2503.00540v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free separation of video encoding from query-time KV retrieval.

**Mechanism:** Sliding-window video encoding; persistent KV stored in RAM/disk; query-conditioned internal/external retrieval (Appendix A p14).

**Evaluation:** Streaming and long VideoQA; efficiency/cache sizing; experiments with multiple video LLMs.

**Author limitations/observations:** p16 future work: offloaded cache growth becomes unsustainable on very long streams. p16: fixed frame block size can disrupt semantic continuity; fixed retrieval count needs dynamic alternatives. p16: precise temporal benchmarks remain limited.

**Analyst inferences:** Analyst inference: Bounded active GPU memory is different from bounded RAM/disk and transfer work.

**Candidate gaps:** Published limitation: total-memory bounded retrieval with stable episodic accuracy. Comparative unresolved: query-adaptive retrieval under equal auxiliary-memory and latency budgets.

**Already occupied / partly addressed:** StreamKV partly addresses semantic segmentation/compression; SAVEMem adapts retrieval scope and bounded visual memory; these old gaps are not untouched.

**Cheap test:** Plot GPU/RAM/disk bytes and query latency versus stream duration using the same retained evidence budget.

#### InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding

File: `2506.15745v2.pdf`; 24 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2506.15745v2).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free length-independent streaming cache compression.

**Mechanism:** Temporal-axis redundancy (TaR), value-norm (VaN) importance and pooling.

**Evaluation:** Four MLLMs / four benchmarks; Jetson AGX Orin real-time memory, speed and energy analysis p10.

**Author limitations/observations:** p24 Appendix H: primarily vision tokens; unified speech/text/video compression left open. p24: TaR/VaN allocation is fixed and could adapt to scene characteristics. p24: learned continual-compression importance remains future work.

**Analyst inferences:** Analyst inference: Information age and repeated-eviction damage need dedicated tests beyond aggregate benchmark averages.

**Candidate gaps:** Published limitation: adaptive spatial/temporal allocation, but novelty requires stronger comparison than a new heuristic. Comparative unresolved: retention stability and old rare-detail survival at fixed total memory.

**Already occupied / partly addressed:** FluxMem, CurveStream, FreshMem and SAVEMem already pursue adaptive/hierarchical memory; energy evaluation is already present here.

**Cheap test:** Compare FIFO, InfiniPot-V and corrected HERMES on identical long prefixes, budgets and evidence-lag bins.

#### Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities

File: `2507.06261v6.pdf`; 73 PDF pages; relevance: **peripheral**. [Primary source](https://arxiv.org/abs/2507.06261v6).

Review depth: Targeted skim: abstract, long-video evaluation and appendix sampling setup; not a full 73-page review.

**Contribution:** General multimodal reasoning and long-context Pro/Flash model family.

**Mechanism:** Native long-context multimodal processing; cache policy not available as an open intervention.

**Evaluation:** Broad reasoning, multimodality and agents; video frame caps differ by benchmark (Appendix pp65–66).

**Author limitations/observations:** Specific cache-method limitation not established in targeted sections.

**Analyst inferences:** Analyst inference: Closed model and API preprocessing limits prevent controlled token-level cache ablations on the available rented GPU.

**Candidate gaps:** Transferable evaluation concept: disclose sampling and frame caps; no direct cache-method gap inferred from proprietary model scale.

**Already occupied / partly addressed:** Long-context capability already demonstrated; not equivalent to bounded online memory.

**Cheap test:** Use as optional external reference only; prioritize reproducible open-backbone comparisons.

#### StreamingTOM: Streaming Token Compression for Efficient Video Understanding

File: `2510.18269v2.pdf`; 14 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2510.18269v2).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free coordinated compression before LLM and quantized online KV memory.

**Mechanism:** Causal Temporal Reduction; Online Quantized Memory with 4-bit storage and on-demand retrieval/dequantization.

**Evaluation:** p6: VideoMME, MLVU, EgoSchema; RVS-Ego and RVS-Movie online QA; long-video sampling 0.5/0.2 fps.

**Author limitations/observations:** p14: compression cannot solve backbone fine-grained perception or long-range reasoning limits. p14: vision encoder remains dominant latency source; encoder reduction and lightweight tuning suggested.

**Analyst inferences:** Analyst inference: Quantized historical store and active cache require separate memory accounting.

**Candidate gaps:** Published limitation: improve front-end cost and task-aware allocation with true end-to-end measurement. Comparative unresolved: same latency budget, old evidence fidelity and cumulative approximation damage.

**Already occupied / partly addressed:** Joint pre-LLM reduction + cache compression + quantization is already addressed here.

**Cheap test:** Profile decode, vision encoder, prefill, compression, retrieval and response separately, including live arrival lag.

#### StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

File: `2511.07278v1.pdf`; 10 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2511.07278v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Dynamic semantic segments, summary representations and guidance-prompt compression.

**Mechanism:** Summary-vector retrieval and layer-adaptive KV budgets.

**Evaluation:** p6: LLaVA-OneVision-Qwen2-7B, H20 96GB FP16, 0.5fps, 15K local window, 4–64-frame segments, eight retrieved frames; StreamingBench and long-video tests.

**Author limitations/observations:** p7 measured observation: increasing retrieved frames can lower accuracy by admitting irrelevant context. No dedicated self-limitations section identified in selected sections.

**Analyst inferences:** Analyst inference: Default retrieved-frame count remains fixed despite semantic segmentation; high-memory deployment needs constrained-hardware confirmation.

**Candidate gaps:** Comparative unresolved: selective evidence coverage with query-specific retrieval count and equal total storage.

**Already occupied / partly addressed:** Semantic segments and layer allocation are already addressed; SAVEMem also adapts retrieval scope.

**Cheap test:** Sweep 4/8/16 retrieved frames on paired episodic queries, measure precision and answer accuracy rather than assuming more context helps.

#### Qwen3-VL Technical Report

File: `2511.21631v2.pdf`; 42 PDF pages; relevance: **supporting**. [Primary source](https://arxiv.org/abs/2511.21631v2).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Dense and MoE multimodal models with interleaved MRoPE, DeepStack and textual timestamp alignment.

**Mechanism:** Native 256K interleaved multimodal context, extended context tests.

**Evaluation:** Broad video, image, document, reasoning and grounding tasks; long-context needle evaluation p25.

**Author limitations/observations:** p25 Section 6: interactive perception, tool reasoning, real-time control and unified understanding/generation proposed as future directions.

**Analyst inferences:** Analyst inference: Backbone capability is distinct from original bounded-cache policy; timestamp mechanism differs from earlier Qwen.

**Candidate gaps:** Comparative unresolved: cross-backbone robustness of a proven retention intervention.

**Already occupied / partly addressed:** Timestamp and architecture improvements are already backbone contributions.

**Cheap test:** Only after a useful small-backbone effect, reproduce on one feasible larger open backbone with its native positions.

#### Memory in the Age of AI Agents: A Survey — Forms, Functions and Dynamics

File: `2512.13564v2.pdf`; 107 PDF pages; relevance: **supporting**. [Primary source](https://arxiv.org/abs/2512.13564v2).

Review depth: Targeted skim: taxonomy/abstract and Section 7 agenda pp69–76; not exhaustive 107-page review.

**Contribution:** Taxonomy of memory forms, functions and dynamics; broad research agenda.

**Mechanism:** Token, parametric and latent memory; retrieval versus generation and automated management.

**Evaluation:** Survey synthesis; no controlled streaming VideoQA compression experiment.

**Author limitations/observations:** pp69–76 Section 7: open directions in automated management, multimodal memory, trustworthy memory and learned/shared memory.

**Analyst inferences:** Analyst inference: Agent-memory desiderata cannot establish an unaddressed video KV-cache problem without direct comparisons.

**Candidate gaps:** Transferable concept: memory provenance and evidence auditability; speculative relevance to temporal visual evidence.

**Already occupied / partly addressed:** Adaptive memory and multimodal memory are established broad directions, not standalone originality claims.

**Cheap test:** Tag retained evidence by original timestamp and test whether answers can identify supporting intervals.

#### FreshMem: Brain-Inspired Frequency-Space Hybrid Memory for Streaming Video Understanding

File: `2602.01683v2.pdf`; 28 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2602.01683v2).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Reconciles historical gist with fine detail using frequency and spatial memory; also component-wise fine-tuning.

**Mechanism:** Multi-scale Frequency Memory with residual details; Space Thumbnail Memory with adaptive episodic clustering.

**Evaluation:** StreamingBench, OV-Bench, OVO-Bench; MLVU and MVBench offline; training-free and tuned variants separated.

**Author limitations/observations:** p8 Figure 3 and p9 discussion explicitly show distant frequency reconstructions fading; residual details mitigate this. Checklist p22 points to Section 5 for limitations, but selected Section 5 p10 is a conclusion without a concrete separate limitation list.

**Analyst inferences:** Analyst inference: Do not compare tuned scores to a frozen training-free policy without disclosure. Analyst inference: Reconstruction/thumbnail loss may harm rare text or object details; this is a hypothesis, not a measured failure shown here.

**Candidate gaps:** Comparative unresolved: gist/detail trade-off at large evidence lag with OCR and object identity queries.

**Already occupied / partly addressed:** Hierarchical episodic memory, residual details and adaptive compression already addressed.

**Cheap test:** Stratify old-detail OCR/identity questions and compare memory pictures or selected evidence plus paired QA.

#### Going Down Memory Lane: Scaling Tokens for Video Stream Understanding with Dynamic KV-Cache Memory

File: `2602.18434v1.pdf`; 19 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2602.18434v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Diagnoses later-frame bias with dense tokens and combines adaptive KV selection with retrieval experts.

**Mechanism:** Adaptive compression/selection in sliding windows; mixture of internal KV and CLIP/PECore external retrieval.

**Evaluation:** CG-Bench, LVBench, VideoMME-long; RVS-Ego/Movie; Qwen2.5-VL-7B; p7 17K window and 11.1GB/h stored cache.

**Author limitations/observations:** p7: external retrieval slightly degrades VideoMME, plausibly from holistic task needs. p7: RVS-Movie accuracy drops despite RVS-Ego gains; authors suggest aggressive compression.

**Analyst inferences:** Analyst inference: External experts add GPU/memory cost (Table3 29 to32GB); storage still grows with time.

**Candidate gaps:** Published observation: retrieval/holistic reasoning trade-off. Comparative unresolved: calibrate age/retrieval bias without extra encoder and with bounded total memory.

**Already occupied / partly addressed:** Recency-bias diagnosis, token scaling and retrieval MoE are already addressed.

**Cheap test:** Compute retrieval age distributions and hit rates under equal retained tokens; include holistic versus localized questions.

#### FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

File: `2603.02096v1.pdf`; 11 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2603.02096v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free causal hierarchical scene-adaptive token compression.

**Mechanism:** Temporal Adjacency Selection and Spatial Domain Consolidation; distribution-adaptive thresholds; short/mid/long memory.

**Evaluation:** StreamingBench, OVO-Bench plus VideoMME/MLVU/LongVideoBench; memory and latency; p8 ablates hierarchy.

**Author limitations/observations:** No dedicated author self-limitations passage identified in selected sections.

**Analyst inferences:** Analyst inference: Similarity/variation is not proof of relevance to an unpredictable future query. Analyst inference: Statistical adaptivity may change performance at scene transitions or low-motion semantically important events; untested here.

**Candidate gaps:** Comparative unresolved: rare static evidence and distribution-shift retention under constant token/auxiliary memory.

**Already occupied / partly addressed:** Causal adaptive hierarchy and scene-statistics-driven compression are already addressed.

**Cheap test:** Create held-out low-motion versus high-motion evidence bins and compare retention and QA at matched budgets.

#### CurveStream: Boosting Streaming Video Understanding in MLLMs via Curvature-Aware Hierarchical Visual Memory Management

File: `2603.19571v1.pdf`; 18 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2603.19571v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Geometric semantic-transition scoring under a strict token budget.

**Mechanism:** DINOv2 feature trajectory curvature + first-order change; online K-Sigma thresholds route clear/fuzzy/discard memory.

**Evaluation:** StreamingBench and OVO perception tasks, multiple backbones including Qwen3-VL; offline FAVOR-Bench/MVBench appendix.

**Author limitations/observations:** p9: future work extends to navigation and prolonged robotic manipulation.

**Analyst inferences:** Analyst inference: Additional DINOv2 encoder and adaptive 10–20-frame queue make equal-FPS comparisons different from equal full cost. Analyst inference: Transition intensity can miss static rare evidence; not demonstrated failure without experiment.

**Candidate gaps:** Comparative unresolved: compute-normalized semantic transition versus evidence sufficiency across long delays.

**Already occupied / partly addressed:** Curvature, adaptive thresholding and clear/fuzzy hierarchical retention already addressed.

**Cheap test:** Include DINO cost in end-to-end budget; evaluate low-change OCR and distant episodic queries separately.

#### Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

File: `2603.29252v1.pdf`; 12 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2603.29252v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free iterative KV writing and task-dependent reading, feasible on a single 3090.

**Mechanism:** Dual-path compressed local memory bank and context propagation; attention-based question reading or fast MemIndex.

**Evaluation:** Two video MLLMs; five long-video tasks and OVO backward tracing; p7 single-3090 comparisons.

**Author limitations/observations:** p5 Section 3.3.2: attention/question-encoding memory reading repeats LLM inference for new questions; motivates MemIndex.

**Analyst inferences:** Analyst inference: Constant processing footprint should not be confused with constant total memory-bank size; quantify bank scaling before using that claim. Analyst inference: Question-conditioned offline encoding and query-agnostic online variants require separate treatment.

**Candidate gaps:** Published limitation partly solved: repeated question encoding via MemIndex. Comparative unresolved: reuse history-conditioned semantic states online while keeping scoring faithful after evictions.

**Already occupied / partly addressed:** Memory recall, separate context/local memory, and fast question indexing already addressed.

**Cheap test:** Compare repeated full question encoding against cached scorer with native causal reference and invalidation after eviction.

#### Dive into Claude Code: The Design Space of Today’s and Future AI Agent Systems

File: `2604.14228v2.pdf`; 53 PDF pages; relevance: **peripheral**. [Primary source](https://arxiv.org/abs/2604.14228v2).

Review depth: Targeted skim: abstract, architecture/context discussion and limitations pp36/50; not exhaustive 53-page review.

**Contribution:** Source-level architectural analysis and comparison with OpenClaw and Hermes Agent.

**Mechanism:** Agent loops, permissions, context compaction, session storage; this Hermes Agent is not the HERMES video-cache method.

**Evaluation:** Static source architecture comparisons; not video retention experiments.

**Author limitations/observations:** p50: static snapshot, reverse engineering cannot determine production behavior/design intent; single-system and comparison-snapshot limitations. p36: memoized context may not reflect dynamic changes immediately.

**Analyst inferences:** Analyst inference: Name collision must not be used as evidence about this repository's video method.

**Candidate gaps:** Transferable engineering idea only: explicit cache invalidation and provenance after context changes.

**Already occupied / partly addressed:** Generic memoization/staleness is known software engineering; no video novelty inferred.

**Cheap test:** Use implementation invariant tests for scorer identity/history invalidation; do not add coding-agent baseline.

#### Semantic-Aware Adaptive Visual Memory for Streaming Video Understanding

File: `2605.07897v1.pdf`; 15 PDF pages; relevance: **direct**. [Primary source](https://arxiv.org/abs/2605.07897v1).

Review depth: Targeted full-text review: abstract, method, evaluation tables/setup, conclusion and limitations search; not an exhaustive reproduction audit.

**Contribution:** Training-free semantic memory generation plus adaptive query-time retrieval.

**Mechanism:** Fixed five-question bank, late-interaction MaxSim; short/mid/long memory and spatial coverage; recency gate, selective forgetting (pp4–5).

**Evaluation:** Qwen2.5-VL 3B/7B, OVO-Bench, StreamingBench and ODV-Bench; bounded-memory scaling.

**Author limitations/observations:** p15 Appendix A.3: evaluation is pseudo-streaming and does not penalize encoding latency; true streaming evaluation remains future work.

**Analyst inferences:** Analyst inference: A fixed bank can lack future-query coverage; this is an inference requiring ablation. Analyst inference: Its token-level semantic matching is distinct from deep causal probe-state refresh, but full method/code comparison is needed.

**Candidate gaps:** Published limitation: live arrival-aware end-to-end evaluation. Comparative unresolved: causal history-conditioned semantic utility versus fixed semantic embeddings at equal full compute. Speculative original candidate: measured staleness-triggered cached native causal scoring, contingent on exact overlap audit.

**Already occupied / partly addressed:** Generic semantic probes, multi-probe banks, bounded hierarchy and adaptive retrieval already addressed by SAVEMem; LORE-KV/KV² occupy future probes/selective reconstruction.

**Cheap test:** First compare frozen/manual scorer, native causal reference, reusable native scorer and SAVEMem-like fixed bank; measure scorer disagreement, QA and all refresh costs.

### Inferences

No listed candidate is claimed globally novel or guaranteed publishable. Peripheral agent papers supply transferable engineering concepts, not direct video-cache gaps.

### Gaps

No exact-content duplicate was identified in this assigned slice. SHA256 values permit the parent to check duplicates across the whole library. Reproduction, code-level overlap and held-out experiments remain separate work.
