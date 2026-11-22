"""Freeze the first annotation-blind TwinPast development notebook."""
import ast
import hashlib
import json

from validate_twinpast_fixed_recent_evidence import ARCHIVE, ROOT, TEMPLATE, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-twinpast-automatic.ipynb"
AUDIT = ROOT / "evidence/audits/twinpast_fixed_recent_2026-11-23_audit.json"
INTRO = """# TwinPast automatic history selection on 7B

Upload this notebook alone to the **same AutoDL RTX 4090 instance**. Choose **Restart Kernel
→ Run All**. It can reuse the pinned 7B weights, Python environment and video cache.

Each of the seven development pairs is evaluated branch by branch. The selector sees
that branch's causal prefix and question. Gold answers, annotated evidence, and the other
branch are unavailable to it. The frame pool uses a fixed 0.5 FPS grid. It contains up to
13 sampled older frames and the last four recent ones. Up to six older frames are presented
to the decoder. The oldest candidate pool is capped before inference by uniform coverage.

The experiment compares recent-only, latest older frames, uniformly spaced older frames,
and query-relevance retrieval. All history arms share the same candidate access, final
frame count and recent frames. Relevance uses two-frame probes with both yes/no label
orders. Direct decoding and both correction rules use the same retrieved frames.

The correction rules start from the recent answer. The margin-only rule accepts a new
top option if its advantage over the recent answer grows by at least 0.5 logits when
retrieved history is added. The history-control rule also requires that this advantage
beat the matched uniform-history input by at least 0.5 logits, with a distinct history.
Answer options are scored under both label orders. Every method answers every branch.
Gold answers are loaded after all predictions have been checkpointed.

This is a development screen. The primary mechanism comparison is the history-control
rule versus direct decoding of the same retrieved frames. Judge it by robust paired
answers, recoveries, regressions and compute cost. Confirmation requires fresh sources.

When you see **TWINPAST_AUTOMATIC_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_automatic_evidence.zip**.
"""


def notebook():
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    audit = json.loads(AUDIT.read_bytes())
    assert archive_hash == audit["archive_sha256"] and audit["development_criterion_met"]
    evidence = verify_inventory(ARCHIVE)
    prior = json.loads(evidence["provenance/fixed_recent_cases.json"])
    bundle = json.loads(evidence["provenance/bundle.json"])
    media = {row["pair_id"]: row for row in bundle["media"]}
    plans = {row["pair_id"]: row for row in bundle["manifest"]["selected_pairs"]}
    cases, media_plan, targets = [], [], {}
    for case in prior["cases"]:
        annotation = case["annotation"]
        pair = annotation["pair_id"]
        plan, asset = plans[pair], media[pair]["decoded"]
        cases.append(dict(pair_id=pair, query_s=annotation["t_query"], question=annotation["question"],
                          options=annotation["options"], recent_hashes=asset["a"]["decoded_rgb_sha256"][-4:]))
        media_plan.append(({"pair_id": pair, **{f"video_{b}": plan[f"video_{b}"] for b in "ab"}},
                           {"decoded": {b: {k: asset[b][k] for k in ("bytes", "sha256")} for b in "ab"}}, None))
        targets[pair] = dict(cohort=case["cohort"], answer_a=annotation["answer_a"], answer_b=annotation["answer_b"],
                             evidence_intervals=annotation["evidence_intervals"])
    assert len(cases) == len(targets) == len(media_plan) == 7
    source = {k: bundle["manifest"][k] for k in ("dataset", "revision")}
    spec = dict(fixed_recent_archive_sha256=archive_hash, model_revision=prior["prior_config"]["model_revision"],
                video_processor=prior["prior_config"]["video_processor"], cases=cases,
                media_source=dict(manifest=source), media_plan=media_plan)
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(index):
        return "".join(cells[index]["source"])

    def put(index, value):
        cells[index]["source"] = value.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(text.encode("utf-8") == evidence[name] for name, text in sources.items())
    for name in ("fixed_recent_cases.json", "bundle.json", "fixed_recent_runner.py"):
        sources.pop(f"provenance/{name}")
    sources["provenance/automatic_cases.json"] = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/evaluation_targets.json"] = json.dumps(targets, ensure_ascii=False, indent=2) + "\n"
    sources["provenance/automatic_runner.py"] = (ROOT / "scripts/autodl_twinpast_automatic.py").read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_fixed_recent", "/hermes_twinpast_automatic")
        .replace("HERMES_TWINPAST_FIXED_RECENT_VALIDATE_ONLY", "HERMES_TWINPAST_AUTOMATIC_VALIDATE_ONLY")
        .replace("provenance/fixed_recent_runner.py", "provenance/automatic_runner.py")
        .replace("7 pairs, 70 fixed-recent scores on 7B", "7 pairs, 6 automatic method outcomes per branch and order"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Run branch-local automatic selection and correction\n")
    put(8, get(8).replace("HERMES_twinpast_fixed_recent_evidence.zip", "HERMES_twinpast_automatic_evidence.zip")
        .replace("'twinpast_fixed_recent.log'", "'twinpast_automatic.log'"))
    for index, cell in enumerate(cells):
        cell["id"] = f"twinpast-automatic-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
