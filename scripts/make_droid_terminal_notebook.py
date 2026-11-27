"""Build the DROID terminal-frame diagnostic with the existing pinned assets."""
import ast
import json

from validate_droid_interference_evidence import ARCHIVE, MANIFEST, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-droid-terminal-context.ipynb"
INTRO = """# DROID terminal-state context diagnostic

Use the same AutoDL GPU as the masking run if available, with at least 22 GiB
VRAM. Upload this notebook beside the existing **DROID_natural_history_frames.zip**
and use **Restart Kernel → Run All**.

The three exact archived terminal images are used in each condition. First
score them alone, then add the four shared recent images, then restore the
three earlier images from the original ten-frame annotated input. Use the
same pinned model, direct scorer and both option orders. Check all 28 archived
ten-frame scores and report pair gains and losses at each context step.

The context lengths and positions differ by design. The result is a capability
diagnostic on seven development pairs, with any confirmation on fresh footage.

Frames are resized from [DROID](https://droid-dataset.github.io/), Khazatsky et al.
and the DROID Dataset Team, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

After **DROID_TERMINAL_CONTEXT_COMPLETE**, download
**HERMES_droid_terminal_context_evidence.zip** and save the executed notebook.
If replay fails after launch, run the final download cell manually.
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    archived = verify_inventory(ARCHIVE)
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == archived[name] for name, src in sources.items())
    spec = json.loads(sources.pop("provenance/droid_interference_cases.json"))
    for name in ("droid_interference_mask_plan.json", "droid_interference_runner.py", "droid_state_masks.py"):
        sources.pop(f"provenance/{name}")
    for key in ("mask_plan_sha256", "mask_fill_rgb", "natural_history_archive_sha256"):
        spec.pop(key)
    scores = {b["pair_id"]: b for b in json.loads(archived["exports/pair_scores.json"])}
    raw = {c["pair_id"]: c for c in json.loads(MANIFEST.read_bytes())["cases"]}
    assert len(raw) == len(scores) == len(spec["cases"]) == 7
    for case in spec["cases"]:
        pair = case["pair_id"]
        case.pop("mask")
        for side, branch in case["branches"].items():
            branch.pop("archived_annotated")
            original = raw[pair]["branches"][side]
            assert branch["frames"] == original["frames"]
            lo, hi = original["terminal_interval"]
            history = original["source_indices"]["annotated"]
            assert len(history) == 6 and all(i < lo for i in history[:3]) and all(lo <= i <= hi for i in history[-3:])
            branch["archived_full"] = {
                order: {k: row["original"][k] for k in ("logits", "prediction", "prompt_sha256", "visual_tokens", "lm_tokens")}
                for order, row in scores[pair]["branches"][side]["answers"].items()
            }
    spec.update(archived_full_zip_sha256=digest(ARCHIVE.read_bytes()),
                archived_full_gpu=json.loads(archived["exports/run_config.json"])["gpu"])
    sources.update({
        "provenance/droid_terminal_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_terminal_runner.py": (ROOT / "scripts/autodl_droid_terminal.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_state_interference", "hermes_droid_terminal_context")
        .replace("HERMES_DROID_STATE_INTERFERENCE_VALIDATE_ONLY", "HERMES_DROID_TERMINAL_CONTEXT_VALIDATE_ONLY")
        .replace("provenance/droid_interference_runner.py", "provenance/droid_terminal_runner.py")
        .replace("7 DROID pairs, 3 matched masking conditions", "7 DROID pairs, 3 context lengths"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Replay the full history and check terminal-state recognition\n")
    put(8, get(8).replace("HERMES_droid_state_interference_evidence.zip", "HERMES_droid_terminal_context_evidence.zip")
        .replace("'droid_state_interference.log'", "'droid_terminal_context.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-terminal-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
