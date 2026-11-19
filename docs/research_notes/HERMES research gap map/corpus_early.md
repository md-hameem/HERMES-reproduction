# HERMES research gap review: early corpus (17 PDFs)

Bounded primary-source review, 19 November 2026. This is a file-level ledger of the first 17 PDFs sorted by case-insensitive filename. The repository-wide review is assembled separately. Every record reports its actual review depth; this is not a claim that all pages or all literature have been exhaustively read. Years in the JSON identify initial arXiv years unless the paper is an ACL proceeding; filenames retain the reviewed version. Unknown author limitations are marked unknown.

## What does each assigned paper establish?

### Takeaway

These papers supply necessary baselines, datasets and controls, but they do not establish that the proposed cached-scorer idea is original. The most direct precursors are HERMES, InfiniPot and VideoLLaMB; benchmark papers also close several tempting evaluation novelty claims.

### Cited Findings

#### MovieNet: A Holistic Dataset for Movie Understanding

File: [2007.10937v1.pdf](../../related_papers/2007.10937v1.pdf); 58 PDF pages. [Primary source](https://arxiv.org/abs/2007.10937).

Review depth: Selective text review: abstract, dataset/task descriptions, §5 p14 and split appendix p47; not all 58 pages read closely.

Contribution: Holistic multimodal movie dataset covering 1,100 movies, with character, scene, description and cinematic annotations (abstract). [Source](https://arxiv.org/abs/2007.10937)

Core mechanism: Dataset and task annotations; not a KV retention method. [Source](https://arxiv.org/abs/2007.10937)

Evaluation: Movie understanding, scene/entity recognition and movie segment retrieval; paper includes extensive dataset/task appendix. [Source](https://arxiv.org/abs/2007.10937)

Author-stated limitations / directions:
- PDF p14 §5: extend movies and annotations; explore further approaches and movie-understanding topics. [Source](https://arxiv.org/abs/2007.10937)

Analyst observations (inferences; separate from author claims):
- Inference: movie-level benchmarks alone do not identify whether a streaming failure arose from eviction, perception or narrative reasoning.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: test an independently sourced movie-domain subset with annotated supporting scene intervals before claiming cross-domain retention gains.

Prior-art boundaries:
- MovieNet already supplies holistic movie annotations and retrieval tasks; movie evaluation itself is not novel. [Source](https://arxiv.org/abs/2007.10937)

Cheap discriminating test: CPU: locate 20 accessible movie questions and annotate support scenes; check video access and dataset terms before planning GPU runs.

#### HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

File: [2026.acl-long.381.pdf](../../related_papers/2026.acl-long.381.pdf); 20 PDF pages. [Primary source](https://aclanthology.org/2026.acl-long.381/).

Review depth: Selective detailed review: motivation, layer policy, scoring and §4 setup, conclusion/limitations p9; local code audit is separate evidence.

Contribution: Training-free layer-specific sensory, semantic and anchor memory policies motivated by attention analysis. [Source](https://aclanthology.org/2026.acl-long.381/)

Core mechanism: Shallow recency, middle generic-query semantic importance, deep frame-anchor retention; paper uses 10%/60%/30% layer partition. [Source](https://aclanthology.org/2026.acl-long.381/)

Evaluation: StreamingBench, OVO and RVS-Ego/movie; offline MVBench, Video-MME and EgoSchema; several LLaVA-OneVision and Qwen2.5-VL sizes (§4). [Source](https://aclanthology.org/2026.acl-long.381/)

Author-stated limitations / directions:
- PDF p9 Limitations: no 72B experiments owing to resources; combinations with orthogonal training-free techniques not investigated. [Source](https://aclanthology.org/2026.acl-long.381/)

Analyst observations (inferences; separate from author claims):
- Inference: importance similarity and retained-token overlap are diagnostic proxies; downstream accuracy must establish whether changing a scorer helps.
- Project implementation gap: current repository scorer behavior must be distinguished from the published policy; see root code audit, not an author claim.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Implementation gap: causally propagated generic-query states and correctly aligned positions need downstream QA validation.
- Hypothesis, novelty unproven: reuse scorer states until refresh improves quality per unit cost; include auxiliary state bytes and refresh overhead.
- Evaluation gap: failures need support-age, evidence-survival and model-capability attribution.

Prior-art boundaries:
- The base hierarchical-memory idea, generic query importance and budget-triggered compression are the original HERMES contribution. [Source](https://aclanthology.org/2026.acl-long.381/)

Cheap discriminating test: Pilot FIFO, current HERMES, corrected scorer and refresh variants on identical 20–30 videos; report paired QA, true peak memory and latency, then decide whether a full run is justified.

Version overlap: HERMES_arxiv.pdf: same title/work, version-level overlap; byte identity not established

#### Ego4D: Around the World in 3,000 Hours of Egocentric Video

File: [2110.07058v3.pdf](../../related_papers/2110.07058v3.pdf); 91 PDF pages. [Primary source](https://arxiv.org/abs/2110.07058).

Review depth: Selective text review: abstract and dataset/task overview; 91-page supplement not exhaustively reviewed.

Contribution: Large-scale egocentric video collection with multiple first-person perception and memory-oriented tasks. [Source](https://arxiv.org/abs/2110.07058)

Core mechanism: Dataset, annotations and task definitions; no bounded-KV policy. [Source](https://arxiv.org/abs/2110.07058)

Evaluation: Multiple egocentric benchmarks, including episodic-memory tasks; paper has a large task-specific supplement. [Source](https://arxiv.org/abs/2110.07058)

Author-stated limitations / directions:
- unknown: no sufficiently verified author limitation passage recorded in this bounded pass.

Analyst observations (inferences; separate from author claims):
- Inference: common data origin can correlate evaluation outcomes across Ego4D-derived benchmarks; they are not automatically independent domain replications.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: audit source-video overlap across chosen egocentric benchmarks and separate memory retrieval from perception failures.

Prior-art boundaries:
- Ego4D already supports episodic-memory evaluation; long-term first-person memory is not a new task. [Source](https://arxiv.org/abs/2110.07058)

Cheap discriminating test: CPU: deduplicate video identifiers across existing manifests; choose a small non-overlapping evaluation subset.

#### EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

File: [2308.09126v1.pdf](../../related_papers/2308.09126v1.pdf); 32 PDF pages. [Primary source](https://arxiv.org/abs/2308.09126).

Review depth: Selective review: abstract, temporal-certificate motivation/method, evaluation overview and datasheet; primary arXiv identity verified online.

Contribution: More than 5,000 human-curated five-option QA pairs on three-minute Ego4D clips; introduces temporal certificate sets. [Source](https://arxiv.org/abs/2308.09126)

Core mechanism: Temporal certificates characterize the intrinsic video support required by a question rather than equating clip length with difficulty. [Source](https://arxiv.org/abs/2308.09126)

Evaluation: Long-form multiple-choice video QA; benchmark creation, human/model evaluation and datasheet. [Source](https://arxiv.org/abs/2308.09126)

Author-stated limitations / directions:
- unknown: no explicit author limitation passage reliably isolated.

Analyst observations (inferences; separate from author claims):
- Inference: clip length and question timestamp are insufficient proxies for the age or duration of evidence a retention policy must preserve.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: stratify failures by annotated oldest/latest necessary evidence and evidence duration, not elapsed query time alone.
- Hypothesis: semantic retention gains should concentrate on questions whose support has aged beyond FIFO capacity.

Prior-art boundaries:
- EgoSchema already explicitly argues video length alone does not measure temporal difficulty; support-based temporal evaluation is prior art. [Source](https://arxiv.org/abs/2308.09126)

Cheap discriminating test: Manually annotate minimal support windows for 20–30 existing failures; compare FIFO/hierarchy outcomes by support age without new training.

#### MVBench: A Comprehensive Multi-modal Video Understanding Benchmark

File: [2311.17005v4.pdf](../../related_papers/2311.17005v4.pdf); 19 PDF pages. [Primary source](https://arxiv.org/abs/2311.17005).

Review depth: Selective review: abstract, task design and evaluation overview; not an exhaustive appendix review.

Contribution: Twenty temporally challenging video tasks, using task definitions and conversion of public annotations. [Source](https://arxiv.org/abs/2311.17005)

Core mechanism: Evaluation tasks designed to require temporal reasoning beyond single-frame understanding. [Source](https://arxiv.org/abs/2311.17005)

Evaluation: Multi-task video understanding and VideoChat2 evaluation. [Source](https://arxiv.org/abs/2311.17005)

Author-stated limitations / directions:
- unknown: no explicit author limitation passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: temporal-task accuracy is a useful control but does not itself isolate bounded streaming eviction.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: ensure a proposed memory improvement preserves short-term action/order ability as well as long-term retrieval.

Prior-art boundaries:
- Temporal reasoning tests and single-frame insufficiency are already core MVBench design goals. [Source](https://arxiv.org/abs/2311.17005)

Cheap discriminating test: Run a small task-stratified subset with matched sampled frames and token budgets; include frame-order perturbations only where labels remain interpretable.

#### VILA: On Pre-training for Visual Language Models

File: [2312.07533v4.pdf](../../related_papers/2312.07533v4.pdf); 16 PDF pages. [Primary source](https://arxiv.org/abs/2312.07533).

Review depth: Selective review: abstract, principal pretraining findings and evaluation overview.

Contribution: Studies interleaved pretraining, language-model updating and text instruction blending for visual language models. [Source](https://arxiv.org/abs/2312.07533)

Core mechanism: Pretraining and instruction tuning rather than inference-time retention. [Source](https://arxiv.org/abs/2312.07533)

Evaluation: Image VLM generalization, in-context learning and related multimodal evaluations. [Source](https://arxiv.org/abs/2312.07533)

Author-stated limitations / directions:
- unknown: no explicit limitation passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: backbone training affects perception and in-context reasoning; a weak backbone can hide gains from memory retention.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: identify questions wrong even with uncompressed/recent support available before attributing errors to eviction.

Prior-art boundaries:
- Better pretraining/interleaved data and retaining text ability are prior work; not a feasible original training project within a few rented GPU days. [Source](https://arxiv.org/abs/2312.07533)

Cheap discriminating test: Compare full/relevant-window input against bounded input on the same small failure sample using an available backbone.

#### MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

File: [2404.06395v3.pdf](../../related_papers/2404.06395v3.pdf); 32 PDF pages. [Primary source](https://arxiv.org/abs/2404.06395).

Review depth: Selective review: abstract, training/scaling overview and explicit limitations p18.

Contribution: Studies 1.2B/2.4B non-embedding-parameter language models and scalable training including warmup-stable-decay learning rates. [Source](https://arxiv.org/abs/2404.06395)

Core mechanism: Training scaling and scheduling; not a video-memory mechanism. [Source](https://arxiv.org/abs/2404.06395)

Evaluation: Small language model evaluation and scaling studies. [Source](https://arxiv.org/abs/2404.06395)

Author-stated limitations / directions:
- PDF p18 Limitations: scaling study was not extended by training an LLM to validate the law; WSD application to LLMs not fully explored. [Source](https://arxiv.org/abs/2404.06395)

Analyst observations (inferences; separate from author claims):
- Inference: language-only scaling findings do not establish video KV retention behavior for a particular multimodal 0.5B backbone.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: separate small-backbone reasoning limits from retention-policy limits using available larger-model pilot controls.

Prior-art boundaries:
- Small-model training efficiency is established work; reproducing scaling training is not the intended low-budget original contribution. [Source](https://arxiv.org/abs/2404.06395)

Cheap discriminating test: Use an existing stronger checkpoint for a tiny oracle-support comparison; do not train a new language model.

#### How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites

File: [2404.16821v2.pdf](../../related_papers/2404.16821v2.pdf); 18 PDF pages. [Primary source](https://arxiv.org/abs/2404.16821).

Review depth: Selective review: abstract, architecture/training overview and evaluation scope.

Contribution: Open-source multimodal suite improves visual-language performance through architecture, data and training recipes. [Source](https://arxiv.org/abs/2404.16821)

Core mechanism: Model training and multimodal representation rather than a fixed-memory streaming policy. [Source](https://arxiv.org/abs/2404.16821)

Evaluation: Broad visual-language capability evaluations. [Source](https://arxiv.org/abs/2404.16821)

Author-stated limitations / directions:
- unknown: no explicit limitation passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: stronger spatial perception and resolution can change apparent memory-policy headroom.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: use oracle evidence and matched-resolution controls before treating every wrong answer as lost memory.

Prior-art boundaries:
- Open multimodal model scaling and data curation are already central to InternVL1.5. [Source](https://arxiv.org/abs/2404.16821)

Cheap discriminating test: Tag OCR, fine spatial detail and answer reasoning errors in the existing both-wrong set; test a few with cropped/oracle evidence.

#### Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis

File: [2405.21075v3.pdf](../../related_papers/2405.21075v3.pdf); 13 PDF pages. [Primary source](https://arxiv.org/abs/2405.21075).

Review depth: Selective review: abstract, benchmark construction, modality/duration table and discussion pp1–3; local version v3 is dated 30 May 2025 although first arXiv submission is 2024.

Contribution: 900 videos and 2,700 manually annotated MCQs across six domains; evaluates short, medium and long videos with subtitle/audio settings. [Source](https://arxiv.org/abs/2405.21075)

Core mechanism: Benchmark modality and duration stratification, not memory management. [Source](https://arxiv.org/abs/2405.21075)

Evaluation: Videos from 11 seconds to one hour; three questions per video; short/medium/long groups of 300 videos (PDF pp2–3). [Source](https://arxiv.org/abs/2405.21075)

Author-stated limitations / directions:
- PDF p3 discussion reports performance declining with increasing video length and proposes better long-context architectures and temporal-reasoning training; this is an empirical observation/future direction, not a formal limitations section. [Source](https://arxiv.org/abs/2405.21075)

Analyst observations (inferences; separate from author claims):
- Inference: duration trends conflate evidence density, sampling, modality availability and eviction unless these are controlled.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: report visual-only and subtitle/audio-enabled settings separately and align frame sampling when comparing memory policies.

Prior-art boundaries:
- Duration and modality-stratified evaluation are already part of Video-MME; adding a long-video benchmark alone is not original. [Source](https://arxiv.org/abs/2405.21075)

Cheap discriminating test: Select a small duration-stratified visual-only subset, retain identical sampling and record support age; add subtitle controls only if supported by the actual pipeline.

#### VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs

File: [2406.07476v3.pdf](../../related_papers/2406.07476v3.pdf); 29 PDF pages. [Primary source](https://arxiv.org/abs/2406.07476).

Review depth: Selective review: abstract, connector/audio method overview, evaluation descriptions and caption-evaluation passage; 29 pages not exhaustively reviewed.

Contribution: Spatial-temporal convolution connector and audio understanding branch address temporal and multimodal video understanding. [Source](https://arxiv.org/abs/2406.07476)

Core mechanism: Trained connector and audio-video modeling; not training-free visual KV eviction. [Source](https://arxiv.org/abs/2406.07476)

Evaluation: Video QA/captioning and audio-related evaluations; long appendix. [Source](https://arxiv.org/abs/2406.07476)

Author-stated limitations / directions:
- unknown: no explicit author limitations passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: audio-dependent failures cannot be repaired by retaining more visual KV entries alone.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: classify missing modality versus evicted evidence before prioritizing memory interventions.

Prior-art boundaries:
- Audio integration and spatial-temporal connectors already exist; adding audio is not an unoccupied novelty claim. [Source](https://arxiv.org/abs/2406.07476)

Cheap discriminating test: On 20 both-wrong items, label whether answer requires audio/subtitles, tiny visual detail, temporal relation or distant visual support.

#### VideoLLM-online: Online Video Large Language Model for Streaming Video

File: [2406.11816v1.pdf](../../related_papers/2406.11816v1.pdf); 19 PDF pages. [Primary source](https://arxiv.org/abs/2406.11816).

Review depth: Selective review: online setup, method/evaluation overview, domain discussion p8 and explicit Appendix D limitations p19.

Contribution: Online video-language formulation processes an ongoing stream and supports timely responses instead of only end-of-video questions. [Source](https://arxiv.org/abs/2406.11816)

Core mechanism: Streaming dialogue training and response scheduling; learned method rather than a drop-in KV policy. [Source](https://arxiv.org/abs/2406.11816)

Evaluation: Streaming video dialogue, including activity-oriented first-person settings. [Source](https://arxiv.org/abs/2406.11816)

Author-stated limitations / directions:
- PDF p19 Appendix D: inadequate high-quality streaming dialogue data limits generalization; generated dialogues miss varied real-world requirements; small-data overfitting; spatial ability weak with fewer spatial tokens. [Source](https://arxiv.org/abs/2406.11816)

Analyst observations (inferences; separate from author claims):
- Inference: replayed timestamped QA does not establish asynchronous real-time interaction, silent periods or deadline behavior.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: measure update throughput and query response latency under a simple arrival schedule, in addition to offline QA.
- Evaluation gap: spatial detail versus temporal capacity needs matched-token controls.

Prior-art boundaries:
- Online video dialogue and response scheduling are already established here; they cannot alone be claimed as a new contribution. [Source](https://arxiv.org/abs/2406.11816)

Cheap discriminating test: Replay 2–3 existing videos with fixed wall-clock frame arrival and injected questions; log backlog, missed deadlines and peak memory.

#### LLaVA-OneVision: Easy Visual Task Transfer

File: [2408.03326v3.pdf](../../related_papers/2408.03326v3.pdf); 43 PDF pages. [Primary source](https://arxiv.org/abs/2408.03326).

Review depth: Selective review: abstract, model/task-transfer setup and video evaluation overview; 43-page supplement not exhaustive.

Contribution: Consolidated single-image, multi-image and video model with transfer between visual task settings. [Source](https://arxiv.org/abs/2408.03326)

Core mechanism: Shared trained visual-language model; not a streaming retention policy. [Source](https://arxiv.org/abs/2408.03326)

Evaluation: Broad visual tasks across image/multi-image/video; extensive supplementary results. [Source](https://arxiv.org/abs/2408.03326)

Author-stated limitations / directions:
- unknown: no explicit author limitation passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: general video transfer does not guarantee stable attention-scoring semantics under repeated eviction and cache compaction.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Implementation/evaluation gap: validate scorer changes on the actual backbone and positions used rather than extrapolating a mechanistic observation across families.

Prior-art boundaries:
- Cross-task visual transfer and this backbone are prior work; HERMES already evaluates LLaVA-OneVision variants. [Source](https://arxiv.org/abs/2408.03326)

Cheap discriminating test: Log a small prefix with actual backbone attention and compare selector decisions plus paired QA; count all auxiliary memory.

#### Video-CCAM: Enhancing Video-Language Understanding with Causal Cross-Attention Masks for Short and Long Videos

File: [2408.14023v1.pdf](../../related_papers/2408.14023v1.pdf); 14 PDF pages. [Primary source](https://arxiv.org/abs/2408.14023).

Review depth: Selective review: abstract, projector/masking method, length-generalization description and conclusion p10.

Contribution: Causal cross-attention masking in a trained video projector improves temporal consistency and length generalization. [Source](https://arxiv.org/abs/2408.14023)

Core mechanism: Causally masked cross-attention projector, distinct from HERMES generic-query scorer masks. [Source](https://arxiv.org/abs/2408.14023)

Evaluation: 4B/9B/14B models; training with 16 frames and evaluation extending to 96 frames. [Source](https://arxiv.org/abs/2408.14023)

Author-stated limitations / directions:
- unknown: no explicit author limitation passage verified; PDF p10 conclusion summarizes causal temporal projector.

Analyst observations (inferences; separate from author claims):
- Inference: causality must be specified at the exact layer and operation; a causal projector result is not proof that a custom KV scorer is correct.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Implementation gap: test the repository scorer with allowed causal history and proper positions; evaluate QA rather than assuming a mask is sufficient.

Prior-art boundaries:
- Causal cross-attention for temporal consistency already exists here; causality in itself is not new. [Source](https://arxiv.org/abs/2408.14023)

Cheap discriminating test: Unit-level attention inspection on one short prefix plus downstream small QA pilot with causal versus current scorer under equal budget.

#### Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input

File: [2408.15542v1.pdf](../../related_papers/2408.15542v1.pdf); 13 PDF pages. [Primary source](https://arxiv.org/abs/2408.15542).

Review depth: Selective review: abstract, curriculum/data overview and evaluation framing.

Contribution: Data curation and curriculum improve a video-language model supporting longer video input. [Source](https://arxiv.org/abs/2408.15542)

Core mechanism: Training curriculum increases input frames/resolution; not a bounded unknown-future-query KV policy. [Source](https://arxiv.org/abs/2408.15542)

Evaluation: Long-context video-language evaluation for an 8B-scale model. [Source](https://arxiv.org/abs/2408.15542)

Author-stated limitations / directions:
- unknown: no explicit author limitation passage verified.

Analyst observations (inferences; separate from author claims):
- Inference: more input capacity or better temporal training does not establish a fair fixed-memory retention comparison.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: equalize actual visual evidence, frame sampling and total memory when comparing a bounded policy with a long-context model.

Prior-art boundaries:
- Long-video curricula and increased frame support are prior approaches, requiring training outside the immediate low-budget plan. [Source](https://arxiv.org/abs/2408.15542)

Cheap discriminating test: Use available model checkpoints only as a small oracle/context control; report differing compute and do not call unmatched models retention-policy baselines.

#### VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges

File: [2409.01071v2.pdf](../../related_papers/2409.01071v2.pdf); 15 PDF pages. [Primary source](https://arxiv.org/abs/2409.01071).

Review depth: Selective review: abstract, memory/SceneTiling method, §3.6 p6, conclusion p8 and training appendix p13; primary arXiv identity/version verified online (v2 revised 2025).

Contribution: Memory bridge layers and temporal memory tokens encode streaming video; SceneTiling segments semantic units. [Source](https://arxiv.org/abs/2409.01071)

Core mechanism: Learned recurrent memory bridge plus semantic segmentation; distinguish trained bridge from training-free use of SceneTiling. [Source](https://arxiv.org/abs/2409.01071)

Evaluation: VideoQA, egocentric planning and Needle in a Video Haystack (NIAVH); synthetic one-second needles across depth/length at 1 fps (PDF p6 §3.6). [Source](https://arxiv.org/abs/2409.01071)

Author-stated limitations / directions:
- PDF p8 conclusion: further integration of LLM memory with memory bridges is a future direction, not a formal limitation claim. [Source](https://arxiv.org/abs/2409.01071)

Analyst observations (inferences; separate from author claims):
- Inference: a synthetic one-second needle can test retention but does not by itself cover natural distributed evidence, causal relations or low-FPS missed events.
- Feasibility observation: Appendix C p13 describes training using four A800 GPUs; full retraining is outside the stated short single-GPU budget.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: extend existing needle controls with natural support windows, multiple jointly necessary events and fixed memory accounting.
- Hypothesis: semantic scene transitions predict when a cached importance state becomes stale; must compare against fixed refresh and random schedules.

Prior-art boundaries:
- Controlled depth/length video needle tests already exist in NIAVH; recurrent memory and semantic scene segmentation are established. [Source](https://arxiv.org/abs/2409.01071)

Cheap discriminating test: CPU annotate support first; small inference-only experiment moving the same visible evidence to different ages while guaranteeing it survives sampling.

#### InfiniPot: Infinite Context Processing on Memory-Constrained LLMs

File: [2410.01518v1.pdf](../../related_papers/2410.01518v1.pdf); 15 PDF pages. [Primary source](https://arxiv.org/abs/2410.01518).

Review depth: Selective detailed review: abstract, §3 mechanism, evaluation setup/prompt ablation and §6 limitations; primary arXiv identity verified online.

Contribution: Continual Context Distillation retains context within fixed memory without further training. [Source](https://arxiv.org/abs/2410.01518)

Core mechanism: Catalyst Prompt (CaP) injected before cache limit estimates future importance; Novelty under Compression (NuC) emphasizes novel information (PDF p2). [Source](https://arxiv.org/abs/2410.01518)

Evaluation: Text LongBench and needle-in-haystack; context length used as KV-memory proxy; prompt variants include general and question-aware prompts (§4). [Source](https://arxiv.org/abs/2410.01518)

Author-stated limitations / directions:
- PDF p9 §6: predefined compression ratio may be suboptimal; very long-term dependencies not exhaustively tested; actual on-device deployment not evaluated. [Source](https://arxiv.org/abs/2410.01518)

Analyst observations (inferences; separate from author claims):
- Inference: prompt-guided unknown-future relevance predates HERMES; text evidence does not establish video-stream scorer freshness or compute gains.
- Inference: actual bytes/latency may differ from token-count proxies when a method carries extra query states or compressed representations.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Paper-stated gap: investigate adaptive compression, but novelty is now crowded and must be checked against recent work.
- Hypothesis, novelty unproven: adaptive refresh of a propagated generic-query state may improve net QA per latency under video changes.
- Evaluation gap: include auxiliary state, peak allocated/reserved memory and refresh work, not KV entries alone.

Prior-art boundaries:
- CaP already uses temporary generic prompts for future-unknown KV importance; novelty also enters selection via NuC. [Source](https://arxiv.org/abs/2410.01518)
- Adaptive compression is an explicitly proposed direction, not evidence it remains unsolved in 2026. [Source](https://arxiv.org/abs/2410.01518)

Cheap discriminating test: Compare current prompt scoring, causal propagated scoring, fixed-interval reuse and adaptive reuse on identical videos; include refresh-only cost and a random schedule with matched refresh count.

#### StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding

File: [2411.03628v1.pdf](../../related_papers/2411.03628v1.pdf); 28 PDF pages. [Primary source](https://arxiv.org/abs/2411.03628).

Review depth: Selective review: abstract, taxonomy/construction pp1–5 and evaluation framing; primary arXiv identity verified online; appendix not exhaustively reviewed.

Contribution: 18 tasks, 900 videos and 4,500 human-curated questions; five questions at different timestamps per video. [Source](https://arxiv.org/abs/2411.03628)

Core mechanism: Taxonomy of real-time visual, omni-source and contextual understanding; timestamped queries rather than a retention algorithm. [Source](https://arxiv.org/abs/2411.03628)

Evaluation: 13 proprietary/open models in original paper; real-time, audio/video and conversational-context tasks. [Source](https://arxiv.org/abs/2411.03628)

Author-stated limitations / directions:
- unknown: passages criticizing previous VStream-QA are not author-stated limitations of StreamingBench and must not be mislabeled.

Analyst observations (inferences; separate from author claims):
- Inference: aggregate timestamped MCQ accuracy does not identify the location/age of necessary support or guarantee a live system keeps pace with input.
- Project evidence boundary: local 498-video/2,495-question run is a subset and configuration, not the full 900-video benchmark.

Candidate gaps (hypotheses or evaluation/implementation needs; not certified novelty):
- Evaluation gap: separate visual-only task coverage from omni-source/contextual coverage and state exclusions explicitly.
- Evaluation gap: annotate support windows and dialogue dependency for failed questions; prevent future-frame and future-question leakage.
- Evaluation gap: pair QA scores with processing throughput, response latency and true state memory.

Prior-art boundaries:
- Streaming task taxonomy, audio/contextual questions and timestamped queries already exist; a new benchmark claim requires a distinct validated measurement. [Source](https://arxiv.org/abs/2411.03628)

Cheap discriminating test: Audit dataset/task manifests and 20–30 failures; verify each question sees only its prefix and allowed dialogue, then bootstrap paired differences at video level.

### Inferences

- A useful opportunity is a *measured intervention*: show that a correctly implemented scoring state produces better answers or lower total cost under a fixed budget. Better agreement with a reference attention map alone is insufficient. This is an analyst hypothesis motivated by the scoring and distillation mechanisms, not a finding that a method is novel. [HERMES](https://aclanthology.org/2026.acl-long.381/); [InfiniPot](https://arxiv.org/abs/2410.01518).
- Support age, distributed evidence and event visibility should determine the memory stress conditions. Clip duration alone mixes unrelated difficulties. [EgoSchema](https://arxiv.org/abs/2308.09126); [VideoLLaMB](https://arxiv.org/abs/2409.01071).

### Gaps

- Newer 2025–2026 memory methods need separate overlap review; this early shard cannot establish novelty against them. Consult the combined report and the existing novelty landscape.
- Per-paper unknown limitations mean no explicit passage was verified in this bounded pass; they do not mean the authors claimed no limitations.
- Dataset access, licenses and source-video overlap must be checked against the actual evaluation manifests before choosing a new benchmark.

## Which apparent gaps are already occupied by prior work?

### Takeaway

Avoid pitching generic prompt scoring, video needles, causality, temporal benchmarking or online dialogue as original contributions by themselves. The thesis needs a narrower mechanism and a measured advantage against current alternatives.

### Cited Findings

- InfiniPot injects a Catalyst Prompt before the cache limit and combines attention-guided importance with Novelty under Compression; its fixed ratio, long-dependency validation and on-device testing are explicitly discussed as limitations (§6, PDF p9). [Source](https://arxiv.org/abs/2410.01518).
- VideoLLaMB includes NIAVH depth/length tests and a semantic SceneTiling component, in addition to recurrent memory bridges (§3.6, PDF p6). [Source](https://arxiv.org/abs/2409.01071).
- EgoSchema explicitly introduces temporal certificate sets because video length alone does not measure temporal task difficulty. [Source](https://arxiv.org/abs/2308.09126).
- Video-CCAM already uses causal cross-attention in its projector; that operation differs from the manual generic-query scorer in the project. [Source](https://arxiv.org/abs/2408.14023).
- VideoLLM-online already studies streaming dialogue; StreamingBench already covers real-time visual, omni-source and contextual tasks. [VideoLLM-online](https://arxiv.org/abs/2406.11816); [StreamingBench](https://arxiv.org/abs/2411.03628).

### Inferences

- A staleness-triggered reuse idea should be framed as untested until it is compared with fixed refresh, scene-change refresh, random matched-count refresh and current strongest relevant memory mechanisms. Scene-change triggers may overlap semantic segmentation; generic prompt importance overlaps CaP and HERMES.
- Causal propagation, RoPE/position alignment, compaction alignment and dialogue prompt changes are correctness conditions for a reusable query state. Fixing a correctness defect can be valuable engineering but does not automatically create a research contribution.
- Fixed token count is an incomplete efficiency comparison if a method stores additional query states, prototypes or retrieval buffers. Count bytes and all refresh work.

### Gaps

- Whether a reusable scorer state remains accurate across evictions and dialogue changes is not established by these papers.
- Whether an adaptive schedule helps after matching refresh count and overhead remains an experimental question.
- A publication-grade originality claim requires the combined recent-paper review and a successful mechanism-level ablation, not an absence of a phrase in these 17 PDFs.

## What can be tested within the available single-GPU budget?

### Takeaway

Start with a CPU-supported failure audit and a small inference-only pilot. Expensive sweeps are justified only after the pilot identifies recoverable retention failures and a promising quality/cost effect.

### Cited Findings

- HERMES provides the direct training-free hierarchical baseline and explicitly leaves combinations with other training-free techniques for future work (Limitations, PDF p9). [Source](https://aclanthology.org/2026.acl-long.381/).
- VideoLLM-online identifies data generalization, small-data overfitting and spatial-token tradeoffs as limitations (Appendix D, PDF p19). These are different failure causes from eviction. [Source](https://arxiv.org/abs/2406.11816).
- Video-MME includes subtitle/audio settings and short/medium/long videos; StreamingBench includes omni-source tasks. Visual-only evaluation must state its scope. [Video-MME](https://arxiv.org/abs/2405.21075); [StreamingBench](https://arxiv.org/abs/2411.03628).

### Inferences

1. Sample 20–30 existing both-wrong or discordant cases. Annotate necessary evidence windows, modality, spatial-detail requirement and dialogue dependence. Record evidence missed by sampling separately from eviction.
2. Compare FIFO and current HERMES with a relevant-window/oracle-support condition to estimate recoverable memory headroom. Use a stronger available checkpoint only on a few ambiguous cases to check backbone limits.
3. If the headroom exists, compare current scoring, corrected causal scoring and cached scoring. For cached scoring, test fixed refresh, adaptive refresh and random schedules with matched refresh counts. Keep input sampling, questions, precision and KV budget identical.
4. Measure paired answers, video-level uncertainty, support retention, true peak allocated/reserved GPU memory, auxiliary-state bytes, update throughput and response latency. Record end-to-end cost, including scoring and refreshes.
5. Test earlier/later support placement and distributed evidence only with visibility guaranteed at the chosen frame rate. These controls build on NIAVH and temporal certificates; they are not themselves claims of a new benchmark.

### Gaps

- No experiments were run as part of this literature shard. Expected improvements are hypotheses, not results.
- No venue acceptance is guaranteed by a gap map. The professor still needs to confirm qualifying CCF categories/rankings.
- Exact sample size and which baseline checkpoints fit the rented GPU depend on the pilot effect and actual hardware.
