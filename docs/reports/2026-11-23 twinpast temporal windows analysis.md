# TwinPast temporal endpoint selection result

The endpoint selector **missed the advancement criterion**. All methods had zero robust pairs on the eight new development pairs. Endpoints produced 18/32 correct branch-order answers, versus 17/32 for individual relevance and 13/32 for uniform history. These are descriptive counts over shared pairs and repeated option orders.

[Evidence archive](../../evidence/archives/HERMES_twinpast_temporal_windows_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-temporal-windows.ipynb) · [Audit](../../evidence/audits/twinpast_temporal_windows_2026-11-23_audit.json)

| Method | Both branches correct, canonical | Both branches correct, reversed | Both orders correct | Correct branch-order answers |
| --- | ---: | ---: | ---: | ---: |
| Recent only | 0/8 | 0/8 | 0/8 | 16/32 |
| Uniform history | 0/8 | 0/8 | 0/8 | 13/32 |
| Individual relevance | 1/8 | 1/8 | 0/8 | 17/32 |
| Temporal endpoints | 2/8 | 1/8 | 0/8 | 18/32 |

The recent-only pair result follows from shared input and differing correct labels. The endpoint successes in one order did not repeat in the other. Endpoints and individual ranking selected identical final frames on eight of sixteen branches. Both used the same scored windows. The audit verified 26 ZIP payloads, the frozen notebook and evidence manifests, causal frame traces, the window allocation, all answer choices and 496 forward-call records. The evaluation shared 384 relevance probes between the two score-based methods. The summed probe forward time was 21.4 seconds. This excludes other work and does not estimate whole-system latency.

The fixed candidate selection excluded all 12 previously inspected pair IDs. Eight new pairs passed the sampled media-overlap screen and exact recent-frame equality check. The dataset lacks authoritative source IDs, so the screen has unresolved footage provenance.

## Evidence coverage and next decision

The sub-second check has completed and recovered two robust pairs. [Sub-second audit and follow-up](<2026-11-23 twinpast subsecond analysis.md>)

The full 1 FPS history pool intersected the annotated evidence in **8/14** branches with interval annotations. Six annotated branches had no history candidate within their intervals. The remaining two of sixteen branches lacked annotations. Endpoint and individual selection each retained at least one interval-matched frame on **5/14** annotated branches. Uniform retained one on **0/14**. This is a timestamp-overlap check. It leaves open whether the selected pixels contain all decisive evidence.

**Next experiment.** Run one annotated-evidence capability check on the seven pairs with evidence intervals. Decode actual frame timestamps within the intervals, including sub-second intervals that 1 FPS missed, and keep the six-older plus four-recent answer budget. Use chronological coverage within the intervals and fill remaining slots from past footage. Preserve both option orders, shared recent frames and the 7B checkpoint. Use same-budget uniform as the reference and report the pair-level change against the archived results. Inspect the selected frames for visible evidence before interpreting the outcome.

The [sub-second AutoDL notebook](../../notebooks/autodl/hermes-twinpast-subsecond.ipynb) uses archived uniform frames to fill any unused history slots and exports a contact sheet for each arm. Its development rule requires at least two robust recoveries and successful uniform replay before advancing to a new sampler.

If this guided control recovers robust pairs, test evidence-aware sampling against dense uniform on new development footage, and keep the endpoint selector retired at its tested setting. If it cannot recover robust pairs, audit visual sufficiency and model capability before committing to another selector. A validated original method and independent confirmation remain open.
