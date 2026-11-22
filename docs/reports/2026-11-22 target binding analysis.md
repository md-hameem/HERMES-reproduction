# Target binding results and research decision

The frozen AutoDL run completed on an RTX 4090. The audit verified 395 payload files, the executed notebook's embedded runner and final ZIP hash, and 300 question checkpoints containing 1,295 distinct window answers and 1,800 selector outcomes. Reconstructed tables, paired statistics and review exports match the saved files. The 30 min 48 s post-model phase includes GPU feature preparation and inference. Download time is outside that measurement. [Executed notebook](../../notebooks/executed/2026-11-22_hermes-target-binding.ipynb) · [Audit](../../evidence/audits/target_binding_2026-11-22_audit.json)

## Fixed-budget result

| Selector | Correct / 269 | Median component time |
|---|---:|---:|
| Recent | 156 | 248.2 ms |
| Previous | 141 | 246.0 ms |
| Global relevance | 147 | 674.7 ms |
| Target only | 150 | 670.0 ms |
| Mean target and evidence score | 152 | 920.9 ms |
| Minimum target and evidence score | 147 | 920.1 ms |

The primary minimum-versus-mean estimate was −1.86 percentage points (95% video-bootstrap interval −4.40 to +0.37). The two rules chose different windows 64 times. The minimum rule won twice and lost seven times on correctness. Omitting any one clip kept the point estimate negative. Against recent-only it gained 14 answers and lost 23, a net −3.35 points (95% interval −7.64 to +0.74). It tied global relevance on aggregate correctness. On the secondary 23-question text-rich slice, recent and previous each scored 13, mean scored 11, and minimum scored 10. This small slice is exploratory. Component times sum the required cache builds, probes and one answer from the shared replay.

## What the logged windows establish

| Correctness available across windows | Questions |
|---|---:|
| Recent and at least one older window | 142 |
| Older window only | 33 |
| Recent window only | 14 |
| No correct candidate window | 80 |

A gold-assisted selector would score 189/269 (70.26%), 12.27 points above recent-only. That ceiling comes from observed answers. It cannot identify whether a correct answer used visible evidence or whether a wrong answer followed from missing evidence. The minimum-score rule missed 19 of the 33 questions that only an older-window answer could solve, while losing 23 correct recent answers. The candidate method misses the predeclared advancement gate. Preserve this result and keep recent-only as the working baseline.

## Next research step

Review the 64 distinct mean/minimum window pairs using the archived blinded worksheet, with a separate key. Also inspect a fixed sample from the 33 older-only, 14 recent-only, and 80 all-window failure strata. Label actual target support, wrong target or setting, unreadable evidence, and uncertainty before viewing method names and correctness. Use the original sampled frames for fine text. The result can distinguish selection errors from missing evidence and answering errors, and determine whether a revised method merits fresh evaluation. This review can use the existing media without a new GPU sweep.

Evidence comes from one backbone and a frozen subset within StreamingBench. Original source overlap is unverified. The archive contains checksums and timestamp records for the underlying media and features. A replay of their content requires the external assets. [Frozen protocol](../../paper/neurocomputing/target_binding_protocol.md) · [Original evidence ZIP](../../evidence/archives/HERMES_target_binding_evidence.zip)
