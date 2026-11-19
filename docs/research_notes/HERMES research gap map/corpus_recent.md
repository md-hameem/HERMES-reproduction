# HERMES research gap map — recent local corpus

Reviewed 2026-11-19. This shard covers the final 17 of 51 local PDF files sorted case-insensitively by filename. All 17 parsed successfully; hashes and page counts are in `corpus_recent.json`. This is a selective method/evaluation/limitations review, not a claim that every line or every possible research gap has been exhausted. One historical reprint is peripheral. HERMES appears as another version of an existing work, so PDF count must not be presented as unique-paper count.

## Takeaway

The broad ideas of object-aware memory, selective historical retrieval, generation-time visual reinjection, delayed-query summary memory, cheap scoring, and adaptive per-layer budgets already have implemented precedents. A defensible contribution needs a narrower causal mechanism and a matched experiment. A paper proposing future work creates a comparison obligation; it does not demonstrate that the proposed problem has been solved.

## Cited findings and per-paper notes

### 35. Towards a Dynamic and Fixed-budget Memory Bank for Efficient Streaming Video Understanding

Local evidence: `docs/related_papers/2606.25658v1.pdf` (16 PDF pages). Identifier: 2606.25658. Relevance: direct. Review depth: abstract; method; setup; conclusion; Appendix A/B targeted full-text review; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free fixed-budget visual memory maintained with an evolving orthonormal semantic basis. Basis-relative novelty/redundancy drives online retention; no future query required.

**Evaluation:** StreamingBench and OVO-Bench; offline VideoMME, LongVideoBench, MLVU, LVBench; LLaVA-OneVision and Qwen2.5-VL.

**Author statement / identified caveat:** Appendix A, PDF p13: frozen-backbone visual perception, temporal reasoning and instruction-following constrain performance; stronger backbones proposed.

**Analyst inference:** Inference: semantic residual energy can be an imperfect proxy for a short, low-energy but task-critical cue. This failure is a hypothesis, not established by the paper.

**Candidate comparative question:** Compare semantic novelty against causal-query fidelity on brief evidence and scene changes at equal budget.

**Existing overlap:** Implemented online semantic-basis novelty already addresses generic semantic-diversity retention.

**Low-cost diagnostic:** Replay a short delayed-cue stream, log cue survival and maintenance cost; avoid a full benchmark until mechanism differs.

Primary source: https://arxiv.org/abs/2606.25658.

### 36. ObjectStream: Latent Objects as Memory Anchors for Streaming Video Understanding

Local evidence: `docs/related_papers/2607.28312v2.pdf` (10 PDF pages). Identifier: 2607.28312. Relevance: direct. Review depth: abstract; object-anchor mechanism; experimental setup p5; conclusion targeted review; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free object-centric memory without external detectors or segmentation. Spatially coherent frozen representations form cross-frame object anchors; preserve persistent histories, transient residuals and a recent window.

**Evaluation:** Qwen2.5-VL-3B/7B; StreamingBench and OVO-Bench; EgoSchema and VideoMME-Long. PDF p5: online 1 fps, at most 256 frames, 64 objects with 8 tokens/object; offline protocol uses larger budgets.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review; do not treat absence of a section as absence of limitations.

**Analyst inference:** Inference: cross-frame association errors and evaluation beyond the reported online frame cap need controlled measurement.

**Candidate comparative question:** Identity-switch and transient-state survival at fixed byte budget; distinguish persistent entities from absence-based scene states.

**Existing overlap:** Implemented object histories and transient-change memory close a generic claim that object-aware bounded memory is missing. MEMO offers a second, externally detected variant.

**Low-cost diagnostic:** Construct/reuse an identity-switch clip and measure retained event evidence and QA before expensive runs.

Primary source: https://arxiv.org/abs/2607.28312.

### 37. StreamFlow: Dynamic Memory Flows for Streaming Video Understanding

Local evidence: `docs/related_papers/2608.10949v1.pdf` (26 PDF pages). Identifier: 2608.10949. Relevance: direct. Review depth: abstract; sections 3.1-3.4; evaluation setup and causal protocol; conclusion targeted review; selective review, not all lines/appendices.

**Contribution/mechanism:** Combines pre-encoder dynamics filtering, trained latent memory and generation-time retrieval. Pixel/GOP dynamics filters mid-term frames; latent compressor consolidates history; weak visual reliance during generation triggers injection of historical visual latents.

**Evaluation:** Qwen3.5-9B backbone with trained/LoRA memory compressor and teacher-generated supervision; StreamingBench plus MVBench, MLVU and VideoMME; explicit causal frame/query protocol.

**Author statement / identified caveat:** No dedicated author-limitations statement identified in selected sections.

**Analyst inference:** Inference: training data and compressor training make full reproduction a poor default for one rented GPU for a few days; generation-time retrieval cost and false triggers must be budgeted.

**Candidate comparative question:** Compare calibrated retrieval decisions under matched overhead; any new trigger needs gains beyond existing attention-guided injection.

**Existing overlap:** Implemented generation-time attention-guided historical visual injection directly overlaps generic entropy/attention-triggered evidence reinjection.

**Low-cost diagnostic:** Use checkpoint if available; first measure existing trigger precision on a small set rather than train a compressor.

Primary source: https://arxiv.org/abs/2608.10949.

### 38. StreamTTT: Reconciling Real-Time Perception and Long-Term Memory in Streaming VLMs

Local evidence: `docs/related_papers/2608.13416v4.pdf` (18 PDF pages). Identifier: 2608.13416. Relevance: direct. Review depth: abstract; introduction mechanism and training; benchmark table/protocol targeted review; selective review, not all lines/appendices.

**Contribution/mechanism:** Separates long history from attention context through online-updated fast weights. Gated test-time-training memory branch stores historical state; short sliding KV serves present perception; jointly trained on offline and real-time QA.

**Evaluation:** StreamTTT-4B; OVO-Bench perception/backward tracing and StreamingBench RTVU; comparisons explicitly under each reported input protocol.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: training and architecture changes exceed a simple drop-in cache patch; per-paper protocol differences prevent treating headline scores as matched evidence.

**Candidate comparative question:** Training-free approximation of the recent/long-memory tradeoff needs direct controls; architecture equivalence cannot be assumed.

**Existing overlap:** Implemented fast-weight/short-KV separation closes broad claims that separating recent perception from long memory is new.

**Low-cost diagnostic:** Only compare with released weights under matching input/timestamp conditions; gate on memory and inference compatibility.

Primary source: https://arxiv.org/abs/2608.13416.

### 39. Dynamic Hub-and-Spoke Memory for Streaming Video Understanding

Local evidence: `docs/related_papers/2608.30294v1.pdf` (21 PDF pages). Identifier: 2608.30294. Relevance: direct. Review depth: abstract; memory/retrieval description; section 7 author limitations p10; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free entity-linked textual distant memory with recent visual evidence. Typed observations form entity hubs and evidence spokes; query-aware retrieval and graph expansion select historical text.

**Evaluation:** Streaming and long-video benchmarks with frozen VLM backbones; distant history represented textually.

**Author statement / identified caveat:** Section 7, PDF p10: entity recognition/OCR/action errors can propagate into memory; text can lose fine-grained distant detail; robust entity linking left for future work.

**Analyst inference:** Inference: author-admitted propagated observation error supports a reliability/provenance diagnostic, but does not establish that a proposed remedy is novel.

**Candidate comparative question:** How to prevent incorrect generated memory from steering future selection while preserving useful historical conditioning?

**Existing overlap:** Implemented entity-graph retrieval overlaps generic structured memory. Robust entity linking is proposed future work, not an implemented solution.

**Low-cost diagnostic:** Inject one controlled wrong textual observation and compare downstream retrieval/evidence survival with history masked.

Primary source: https://arxiv.org/abs/2608.30294.

### 40. KeyRec: Bounded Visual Memory for Streaming and Long-Video Understanding

Local evidence: `docs/related_papers/2609.32182v1.pdf` (21 PDF pages). Identifier: 2609.32182. Relevance: direct. Review depth: abstract; add-merge-evict/router mechanism; Appendix E limitations p17; selective review, not all lines/appendices.

**Contribution/mechanism:** Separates query-agnostic construction from query-adaptive fixed-budget visual readout. Recent cache plus novel event proposals, similarity/time consolidation and add-merge-evict; text-only query router allocates recent/event read budget.

**Evaluation:** Four benchmarks and three backbones including Gemma 4 E2B/E4B and NEO-ov 2B; post-encoder visual memory and decoder cost.

**Author statement / identified caveat:** Appendix E, PDF p17: fixed update/routing rules; no ViT encoding savings; attempted intermediate token reduction disrupted optimized path with no wall-time gain; lacks explicit identity/trajectory/state modeling.

**Analyst inference:** Inference: combining encoding compression with memory must be measured end to end; lower token count alone does not establish speed.

**Candidate comparative question:** Can native-path-compatible maintenance reduce wall time and retain brief evidence at constant budget?

**Existing overlap:** Implemented bounded event organization and query router already exist. ObjectStream/MEMO implement some object directions merely proposed by KeyRec; learning update rules remains future work here.

**Low-cost diagnostic:** Profile encoder, decoder, update and decode costs separately; compare one compatible encoder optimization on identical clips.

Primary source: https://arxiv.org/abs/2609.32182.

### 41. MEMO: Multi-Level Entity-Aware Memory for Streaming Video Understanding

Local evidence: `docs/related_papers/2609.38900v1.pdf` (10 PDF pages). Identifier: 2609.38900. Relevance: direct. Review depth: abstract; memory mechanism; experimental settings; section 5 limitations p8; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free multi-level global/entity/spatial memory. Detection/segmentation/association inform coherent chunks; lightweight indices retrieve high-resolution visual evidence from a separate CPU pool.

**Evaluation:** StreamingBench and OVO-Bench; LLaVA-OV-0.5B/7B, Qwen2.5-VL-7B and Qwen3-VL-8B; 1 fps; top3 chunks; 16 frame visual input split recent/retrieved.

**Author statement / identified caveat:** Section 5, PDF p8: upstream detector/segmenter/association errors propagate; implicit/absence-based states poorly represented; persistent cross-chunk identity and hour/day consolidation remain open; explicit scene-state memory/reranking/consolidation future work.

**Analyst inference:** Inference: bounded model input is distinct from bounded total CPU/raw-evidence storage; require growth accounting before claiming all-memory bounds.

**Candidate comparative question:** Absence/state-transition evidence and persistent error correction are specific comparative questions; hour/day consolidation is high-cost for this project.

**Existing overlap:** Implemented entity-aware chunk retrieval and object dynamics overlap generic object-memory claims. Explicit absence/state memory is stated future work, not solved by that statement.

**Low-cost diagnostic:** Small clips with object leaving, occlusion and true absence; compare with an entity-memory baseline and report false absence.

Primary source: https://arxiv.org/abs/2609.38900.

### 42. ReMem: Streaming Video Understanding With Long Context Retention

Local evidence: `docs/related_papers/2610.05940v1.pdf` (16 PDF pages). Identifier: 2610.05940. Relevance: direct. Review depth: abstract; SCM/RVM and GQC method pp3-4; experiments p5; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free adaptation combines fixed-length context state and question-conditioned vision retrieval. SCM recursively compresses previous context plus new visual tokens using GQC; RVM selects question-relevant tokens by cosine similarity and reinjects them into context.

**Evaluation:** LLaVA-OV-7B and Qwen2.5-VL-7B; StreamingBench contextual SQA/MCU/ACU, OVO EPM/ASI/HLD; EgoSchema and VideoMME.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: repeated compression can create irreversible loss, and fixed SCM size alone does not prove bounded whole-system RVM storage.

**Candidate comparative question:** Compression-depth/cue-survival stress test with matched complete memory accounting.

**Existing overlap:** Implemented fixed compressed context and query-driven visual reinjection overlap generic recall/reinjection ideas; ProtoKV separately studies delayed queries.

**Low-cost diagnostic:** Measure answer stability and evidence survival after repeated irrelevant chunks, keeping query semantics fixed.

Primary source: https://arxiv.org/abs/2610.05940.

### 43. Memory: A Contribution to Experimental Psychology

Local evidence: `docs/related_papers/ANS0972-7531-20-155.pdf` (2 PDF pages). Identifier: historical-reprint. Relevance: peripheral. Review depth: two-page extracted text and bibliographic header review; selective review, not all lines/appendices.

**Contribution/mechanism:** Historical account of memory experiments; conceptual context only. Human learning/retention experiments; no computational KV retention mechanism.

**Evaluation:** Two-page 2013 Annals of Neurosciences reprint of historical Ebbinghaus work/translation.

**Author statement / identified caveat:** Not applicable to modern computational-method claims; no transformer/KV limitations claimed.

**Analyst inference:** Inference: a biological analogy supplies motivation but no evidence of machine-learning novelty or algorithmic superiority.

**Candidate comparative question:** No original HERMES method gap can be inferred from this source alone.

**Existing overlap:** Not applicable.

**Low-cost diagnostic:** Not applicable; use empirical computational controls instead.

Primary source: bibliographic identity from local reprint; no online verification in this pass.

### 44. HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

Local evidence: `docs/related_papers/HERMES_arxiv.pdf` (34 PDF pages). Identifier: 2601.14724. Relevance: direct. Review depth: abstract; method architecture; experimental setup p6; targeted limitation search; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free hierarchical cache management motivated by layer-wise attention behavior. Shallow recency, deep anchors, middle transition; cross-layer smoothing and RoPE-aware position reindexing; direct reuse at query arrival.

**Evaluation:** PDF p6: StreamingBench, OVO-Bench, RVS-Ego/Movie; MVBench, VideoMME and EgoSchema; LLaVA-OV and Qwen2.5-VL spanning 0.5B-32B; A800 efficiency tests.

**Author statement / identified caveat:** No dedicated author-limitations section identified in targeted review. Do not attribute local scorer defects to the authors without official-code/version comparison.

**Analyst inference:** Inference: project baseline has an inconclusive 4K advantage and a local proxy/native-attention mismatch; correcting implementation is a validity prerequisite, not original contribution.

**Candidate comparative question:** After correction, test cached causal scoring with refresh controls and error-feedback sensitivity against fixed native scoring.

**Existing overlap:** Same underlying work as local 2026.acl-long.381.pdf (version overlap; not necessarily identical bytes). Hierarchical retention itself is existing HERMES.

**Low-cost diagnostic:** Small paired corrected-scorer pilot with identical memory budget, then method ablations only if benefit survives.

Primary source: https://arxiv.org/abs/2601.14724.

### 45. OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning

Local evidence: `docs/related_papers/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.pdf` (11 PDF pages). Identifier: CVPR2026-OASIS. Relevance: direct. Review depth: abstract; section 3 event-forest formulation; motivation p2 and conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** On-demand event-level temporal routing through multi-resolution event memory. High-fidelity short window, medium buffer, event forest with timestamps/frames/generated summaries/embeddings; short context first and selective distant retrieval.

**Evaluation:** Streaming-video reasoning evaluation; detailed numerical comparability must be checked against actual protocol, not inferred from headline results.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: generated summaries/retrieval may introduce persistent errors; distinguish GPU context bounds from retained frame and text storage.

**Candidate comparative question:** Specific reliability/feedback or calibrated abstention diagnostic may remain comparative; generic uncertainty-triggered history recall is already implemented.

**Existing overlap:** Implemented on-demand short/long event routing overlaps broad EDMI-style historical recall. StreamFlow also implements generation-time visual injection.

**Low-cost diagnostic:** Compare no retrieval, always retrieval and existing selective retrieval on matched context/latency before designing a new trigger.

Primary source: https://openaccess.thecvf.com/content/CVPR2026/html/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.html.

### 46. ProtoKV: Streaming Video Understanding under Delayed Query with Summary-State Memory

Local evidence: `docs/related_papers/minh26a.pdf` (20 PDF pages). Identifier: PMLR-v306-minh26a. Relevance: direct. Review depth: abstract; summary-state mechanism; section 4.1 delayed-query protocol p6; conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** Keeps exact near context and compressed object-centric prototype/residual state for far history. Spatial/semantic prototypes plus residual statistics exposed as bounded pseudo-tokens, rather than selecting only old token instances.

**Evaluation:** Four streaming benchmarks; controlled delayed-query sweeps using dataset timestamp t0; retrospective-type filter and common valid-query intersection (section 4.1 p6).

**Author statement / identified caveat:** Section 4.1, p6 explicitly notes no ground-truth last-evidence time te; delay is measured from dataset query timestamp t0 instead. Conclusion leaves richer summaries/broader streaming settings to future work.

**Analyst inference:** Inference: timestamp-delay sweeps are weaker than annotated last-evidence-delay tests; cue timing needs verification for precise retention claims.

**Candidate comparative question:** Evidence-anchored delay and contamination robustness are more precise than generic delayed-query memory.

**Existing overlap:** Implemented bounded summary state and controlled delayed-query experiments directly overlap broad delayed-evidence retention novelty.

**Low-cost diagnostic:** Use a few clips with known decisive cue time; retrospective answers only; compare matched query sets across delays.

Primary source: https://proceedings.mlr.press/v306/minh26a.html.

### 47. Decouple and Cache: KV Cache Construction for Streaming Video Understanding

Local evidence: `docs/related_papers/pang26a.pdf` (16 PDF pages). Identifier: PMLR-v306-pang26a. Relevance: direct. Review depth: abstract; section 3 and figure 2 p4; conclusion section 6; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free decoupling of cumulative past cache from on-demand recent cache construction. Store pre-RoPE keys; small FIFO feature buffer supplies independently constructed recent/instant cache; combine with cumulative past cache for QA.

**Evaluation:** Streaming-video QA; recent-context construction incurs query-time work; numeric comparisons require matching input and budget.

**Author statement / identified caveat:** Conclusion section 6 proposes faster instant-cache construction and better past-cache optimization as future work.

**Analyst inference:** Inference: pre-RoPE decoupling solves a positional/cache-construction concern but does not itself establish query-state freshness or correctness of local scorer.

**Candidate comparative question:** Amortized causal scoring versus on-demand recent reconstruction is a cost/quality comparison, not proof of novel cache decoupling.

**Existing overlap:** Implemented position-agnostic cache and recent/past separation; related pre-RoPE safety also Video-MemComp.

**Low-cost diagnostic:** Time first query and repeated queries separately with identical recent buffer and total byte budget.

Primary source: https://proceedings.mlr.press/v306/pang26a.html.

### 48. Accelerating Streaming Video Large Language Models via Hierarchical Token Compression

Local evidence: `docs/related_papers/Wang_Accelerating_Streaming_Video_Large_Language_Models_via_Hierarchical_Token_Compression_CVPR_2026_paper.pdf` (11 PDF pages). Identifier: CVPR2026-STC. Relevance: direct. Review depth: abstract; section 3 compressors; contribution list p2; conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free compression spanning vision encoder and LLM prefill. STC-Cacher reuses adjacent-frame visual features; STC-Pruner removes low-saliency tokens while respecting causal unknown-future-query streaming.

**Evaluation:** Streaming VideoLLMs, including ReKV; efficiency and task-quality evaluation.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: high redundancy and end-to-end bottleneck placement determine benefit; report hit rate and wall time, not only token reduction.

**Candidate comparative question:** Can compatible feature reuse help the actual HERMES profile without compounding retention errors?

**Existing overlap:** Implemented causal ViT caching and prefill compression close generic proposals to cache adjacent frames for streaming speed.

**Low-cost diagnostic:** Profile first, then test one encoder-cache setting on identical videos with full wall-clock timing.

Primary source: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Accelerating_Streaming_Video_Large_Language_Models_via_Hierarchical_Token_Compression_CVPR_2026_paper.html.

### 49. MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering

Local evidence: `docs/related_papers/Xiao_MuKV_Multi-Grained_KV_Cache_Compression_for_Long_Streaming_Video_Question-Answering_CVPR_2026_paper.pdf` (11 PDF pages). Identifier: CVPR2026-MuKV. Relevance: direct. Review depth: abstract; compression/retrieval mechanism; section 4.1 p5; conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** Patch-, frame- and segment-grained historical KV compression and efficient retrieval. Self-attention/Fourier compression; offline KV storage; grain-agnostic parallel retrieval then hierarchical reranking at query time.

**Evaluation:** Section 4.1 p5: RVS-Ego, RVS-Movie, StreamingBench RTVU subset; metrics online accuracy, efficiency and offline cache storage; changed GPT-3.5 judge version noted.

**Author statement / identified caveat:** Section 4.1 p5 discloses original GPT-3.5-turbo-0613 judge was deprecated, so a different GPT-3.5-turbo judge is used; this is an evaluation comparability caveat, not a declared algorithm limitation.

**Analyst inference:** Inference: storage/retrieval improves recall but is not equivalent to strictly bounded total-memory eviction; judge-version differences can obscure cross-paper score comparisons.

**Candidate comparative question:** Whole-system boundedness and matched-judge controlled recall comparison.

**Existing overlap:** Implemented multigrain KV retrieval and reranking overlap generic multiresolution-memory recall.

**Low-cost diagnostic:** Report stored bytes versus elapsed stream and use the same answer scorer for any direct comparison.

Primary source: https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_MuKV_Multi-Grained_KV_Cache_Compression_for_Long_Streaming_Video_Question-Answering_CVPR_2026_paper.html.

### 50. Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach

Local evidence: `docs/related_papers/Yang_Revisiting_Multimodal_KV_Cache_Compression_A_Frequency-Domain-Guided_Outlier-KV-Aware_Approach_CVPR_2026_paper.pdf` (11 PDF pages). Identifier: CVPR2026-FlashCache. Relevance: supporting. Review depth: abstract; scoring/budget mechanism; setup and main evaluation table pp5-6; conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** Training-free FlashAttention-compatible multimodal cache compression without attention scores. Frequency-domain outlier key/value signals drive token retention and layer budget assignment.

**Evaluation:** MileBench temporal/semantic multi-image, needle retrieval and image retrieval tasks; LLaVA-OV-1.5-8B, Qwen2.5-VL-7B/32B (section 4.1 pp5-6).

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: offline multimodal evaluation alone does not establish causal unknown-question streaming performance; backbone and policy must be controlled.

**Candidate comparative question:** Compare cheap outlier scoring with causal-state scoring in the same streaming retention policy.

**Existing overlap:** Implemented attention-free scoring and layer-adaptive budget allocation; broad cheap-score/adaptive-layer-budget novelty is occupied.

**Low-cost diagnostic:** Hold budget, policy and selected token granularity fixed; swap only signal on a small streaming replay.

Primary source: https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Revisiting_Multimodal_KV_Cache_Compression_A_Frequency-Domain-Guided_Outlier-KV-Aware_Approach_CVPR_2026_paper.html.

### 51. Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models

Local evidence: `docs/related_papers/Zhang_Think-as-You-See_Streaming_Chain-of-Thought_Reasoning_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf` (11 PDF pages). Identifier: CVPR2026-TaYS. Relevance: supporting. Review depth: abstract; mechanism overview; section 4.1 p6; conclusion; selective review, not all lines/appendices.

**Contribution/mechanism:** Streaming chain-of-thought with separate perception and reasoning caches. Parallel-cache design allows perception and progressive reasoning to evolve while frames arrive; trained on stream-aligned reasoning data.

**Evaluation:** Section 4.1 p6: extended VideoEspresso-derived reasoning protocol; Qwen2.5-VL-3B/7B; batch/interleaved and SFT controls; quality and latency.

**Author statement / identified caveat:** No dedicated author-limitations passage identified in targeted review.

**Analyst inference:** Inference: training-dependent reasoning pipeline is a conceptual comparison, not a drop-in baseline for a few-day cache-only improvement.

**Candidate comparative question:** Separate reasoning correctness from retention quality; wrong intermediate reasoning may influence subsequent state and needs explicit diagnostics.

**Existing overlap:** Implemented progressive streaming reasoning and dual cache structure overlap generic asynchronous reasoning/cache separation.

**Low-cost diagnostic:** If weights available, use one fixed clip and intervene in intermediate reasoning; otherwise keep as conceptual prior art.

Primary source: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Think-as-You-See_Streaming_Chain-of-Thought_Reasoning_for_Large_Vision-Language_Models_CVPR_2026_paper.html.

## Cross-paper inferences and gaps

1. **Causal probe-state reuse and selective refresh — unresolved comparative question.** KV² implements selective reconstruction; LORE-KV implements response-side lookahead states and reliability-weighted utility. Neither abstract establishes online video probe-state reuse with measured staleness control. That observation is not an exhaustive absence proof. Test fixed templates/native scoring versus refreshed causal states versus cached states, then add refresh triggers. Account for all update/query overhead and force refresh when history conditioning changes.

2. **Wrong past-answer feedback — untested project-specific hypothesis.** `inference/llavaov_hermes.py:292` (`predict_next_question`) reads the previous generated question/answer, decodes answer letters into option text, and includes it in both local and global guidance prompts. Thus a wrong previous answer can in principle change later retention scores. This is code-supported dependency, not evidence of actual harm. D-HSM explicitly admits generated-observation error propagation (p10); MEMO admits upstream perception error propagation (p8). LORE-KV reliability weighting addresses sampled future trajectories, a related but different source of uncertainty. No reviewed source here establishes a controlled past-QA-error/eviction experiment. Do not claim universal novelty.

   Low-cost pilot: fork identical cached state before a previous QA; vary only history as original generated answer, corrected past answer, wrong but plausible past answer, and no-answer/question-only history. Use only past ground truth for the corrected diagnostic, never future answers. Freeze all later frames/questions, compare selected-set divergence, decisive-cue survival, next-answer accuracy and total time. Keep future-question information inaccessible during updates. A question-only, confidence-gated or disagreement-triggered refresh policy is a candidate remedy only after measurable harm appears.

3. **Absence and state-transition evidence — specific author-admitted weakness, still needs prior-art narrowing.** MEMO p8 explicitly names absence-based states; KeyRec p17 lacks explicit trajectories/state transitions. ObjectStream already stores object evolution and transient changes, so generic object/state memory is occupied. A meaningful test distinguishes absence from occlusion, departure and detection failure at equal memory. Annotation and tracking costs may exceed available compute time before method benefit is clear.

4. **Repeated compression and true evidence delay — occupied broad direction with a narrower evaluation question.** ProtoKV implements delayed-query summary state and filters retrospective queries; ReMem already targets retention across recursive compression. ProtoKV uses dataset query timestamps because decisive-evidence timestamps are unavailable. Measuring cue-grounded delay with fixed answers, identity switches and irrelevant intervening scenes is a useful diagnostic, not automatically a standalone algorithm contribution.

5. **End-to-end systems efficiency — evidence gap before optimization.** KeyRec explicitly reports encoder pruning that did not improve wall time (p17). STC already caches adjacent-frame encoder features; DeltaS already offers cheap state-drift scoring on hybrid backbones. The local pipeline must be profiled before selecting an optimization; report CPU decode, encoder, LLM, maintenance, retrieval, first-query and repeated-query timing, plus GPU and retained CPU storage growth.

6. **Selective evidence injection — generic direction already implemented.** StreamFlow §3.4, PDF pp5–6 (VAS eq8), injects old visual latents during generation based on weakened visual reliance; OASIS performs selective historical event recall. A new entropy trigger alone needs a clear operational difference, calibrated error/benefit evidence and overhead-matched controls.

7. **Validity prerequisites versus originality.** Fixing the local frozen/manual attention approximation, restoring causal semantics and reproducing native/official behavior are prerequisite engineering. They do not alone supply an original method. Validate the corrected baseline before attributing improvements to a new research component.

## Cost gates for one rented GPU

- First run CPU-only annotation/trace analysis and a small paired inference replay. Do not train StreamFlow, StreamTTT or TaYS from scratch as a first step.
- Advance only if the proposed mechanism changes evidence survival or produces consistent paired QA benefit beyond noise; compare matched byte budgets and retention policies.
- Run one full benchmark only after the corrected baseline and pilot ablations pass; add a distinct task/backbone or targeted evidence-delay set for generalization when affordable.
- Record inconclusive and negative results. A small existing HERMES–FIFO difference does not by itself establish publishable improvement.

## Online follow-up scope

Four recent primary abstract pages were checked on 2026-11-19: LORE-KV v2, KV² v2, Video-MemComp and DeltaS. These are separate records in `online_updates.json` and do not increase local PDF coverage. Their full experimental sections were not independently audited in this pass. SAVEMem and StreamAgent remain necessary cross-shard comparisons for the feedback candidate; no full-text novelty clearance is claimed here.
