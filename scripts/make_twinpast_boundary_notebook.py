"""Build the GPU test of the frozen TwinPast scene-boundary policy."""
import ast
import json

from make_twinpast_readout_notebook import DEST as TEMPLATE
from validate_twinpast_refinement_evidence import ARCHIVE, MANIFEST, ROOT, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-twinpast-boundary.ipynb"
CPU_AUDIT = ROOT / "evidence/audits/twinpast_boundary_cpu_2026-11-27.json"
INTRO = """# TwinPast scene-boundary GPU comparison

Upload this notebook to the same AutoDL GPU container used for refinement and
select **Restart Kernel → Run All**. It reuses the verified Python environment,
pinned LLaVA-OneVision 7B weights and clips when available. Use a GPU with at
least 22 GiB VRAM. On a fresh container, setup can require about 40 GiB of disk.

Compare coarse uniform, local uniform and the frozen boundary-constrained
ranking plus pre-cut anchor on eight development pairs. Each method supplies
six history and four matched recent frames to the original direct scorer.
Both option orders are scored. Recompute question relevance, reproduce the
CPU frame selections and check 64 archived baseline scores before reporting
paired results. Selection and replay mismatches preserve failure evidence.

A development pass requires a robust recovery with no robust regression
versus coarse uniform. The notebook reports per-stage timings, including the
additional boundary scan. The per-policy time is a sum of measured stages
with shared work allocated to each policy. It is not an independent end-to-end
latency benchmark.

After **TWINPAST_BOUNDARY_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_boundary_evidence.zip**. If there is a failure, run the final
download cell manually to retrieve the diagnostic evidence.
"""


def freeze(spec, evidence, report):
    assert report["refinement_archive_sha256"] == spec["source_archive_sha256"] == digest(ARCHIVE.read_bytes())
    assert report["manifest_sha256"] == digest(MANIFEST.read_bytes())
    assert report["summary"]["development_criterion_met"]
    assert report["answer_history_frames"] == 6 and report["shared_recent_frames"] == 4 and report["window_budget"] == 3
    rows = {(r["pair_id"], r["branch"].lower()): r for r in report["branches"]}
    assert len(rows) == 16
    for case in spec["cases"]:
        pair = case["pair_id"]
        previous = json.loads(evidence[f"checkpoints/{pair}.json"])["block"]["branches"]
        for side, archived in case["branches"].items():
            branch, cpu = previous[side], rows[pair, side]
            local, slots = branch["refined"], branch["refined_picks"]["local_uniform"]
            assert [local["indices"][i] for i in slots] == cpu["history_indices"]["local_uniform"]
            assert branch["selected_windows"] == cpu["original_windows"]
            calls = {r["order"]: r for r in branch["calls"] if r["phase"] == "local_uniform"}
            archived["direct"].pop("state_change")
            archived["direct"]["local_uniform"] = {
                order: {**answers["local_uniform"],
                        **{k: calls[order][k] for k in ("prompt_sha256", "lm_tokens", "visual_tokens")}}
                for order, answers in branch["answers"].items()
            }
            archived.pop("rgb")
            archived["local"] = dict(indices=local["indices"], hashes=local["hashes"], slots=slots)
            archived["candidates"] = branch["candidates"]
            archived["boundary"] = dict(original_windows=cpu["original_windows"],
                                        constrained_windows=cpu["constrained_windows"],
                                        indices=cpu["history_indices"]["combined"],
                                        scanned_indices=cpu["decoded_probe_frames"])
    spec.update(cut_threshold=report["threshold"], cut_fps=report["fps"], cpu_audit_sha256=digest(CPU_AUDIT.read_bytes()))
    return spec


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    spec = json.loads(sources.pop("provenance/readout_cases.json"))
    spec = freeze(spec, verify_inventory(ARCHIVE), json.loads(CPU_AUDIT.read_bytes()))
    sources.update({
        "provenance/boundary_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/boundary_runner.py": (ROOT / "scripts/autodl_twinpast_boundary.py").read_text(encoding="utf-8"),
        "provenance/twinpast_boundary_policy.py": (ROOT / "scripts/twinpast_boundary_policy.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_twinpast_readout", "hermes_twinpast_boundary")
        .replace("HERMES_TWINPAST_READOUT_VALIDATE_ONLY", "HERMES_TWINPAST_BOUNDARY_VALIDATE_ONLY")
        .replace("provenance/readout_runner.py", "provenance/boundary_runner.py")
        .replace("8 archived development pairs, 2 fixed frame sets", "8 archived pairs, 3 policies, frozen CPU rule"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Compare the frozen boundary policy with matched baselines\n")
    put(8, get(8).replace("HERMES_twinpast_readout_evidence.zip", "HERMES_twinpast_boundary_evidence.zip")
        .replace("'twinpast_readout.log'", "'twinpast_boundary.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-boundary-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
