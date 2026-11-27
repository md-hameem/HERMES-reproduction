"""Build the DROID terminal-attention notebook from the audited state-swap run."""
import ast
import json

from validate_droid_state_swap_evidence import ARCHIVE, MANIFEST, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-droid-terminal-attention.ipynb"
INTRO = """# DROID fixed-position terminal attention

Use the same AutoDL GPU as the state-swap run, with at least 22 GiB VRAM.
Open this notebook alongside the existing **DROID_natural_history_frames.zip**
and select **Restart Kernel → Run All**.

Replay the seven DROID development pairs with the same ten frames, positions,
questions and answer orders. Compare native attention, an explicit causal
control, direct access to three annotated terminal frames, and direct access
to three earlier frames. Only direct visual attention from the question and
answer scaffold changes. The visual prefix can still carry earlier information
through contextualized representations.

The feasibility criterion requires two pairs recovered only by terminal
access and preservation of all robust causal-control pairs. A robust pair
must pass both branches in both option orders. Both replay gates must pass
before interpreting the results.

Frames are resized from [DROID](https://droid-dataset.github.io/), Khazatsky et al.
and the DROID Dataset Team, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

After **DROID_TERMINAL_ATTENTION_COMPLETE**, save the executed notebook and
download **HERMES_droid_terminal_attention_evidence.zip**. If the runner
fails, run the final download cell to retrieve failure evidence.
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    archived = verify_inventory(ARCHIVE)
    assert json.loads(archived["completion.json"]) == {"status": "complete", "pairs": 7}
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == archived[name] for name, src in sources.items())
    assert archived["provenance/droid_natural_manifest.json"] == MANIFEST.read_bytes()
    spec = json.loads(sources.pop("provenance/droid_swap_cases.json"))
    sources.pop("provenance/droid_swap_plan.json")
    sources.pop("provenance/droid_swap_runner.py")
    spec.pop("swap_plan_sha256")
    config = json.loads(archived["exports/run_config.json"])
    spec.update(prior_archive_sha256=digest(ARCHIVE.read_bytes()), prior_gpu=config["gpu"])
    assert spec["frame_zip_sha256"] == config["frame_zip_sha256"]
    assert spec["model_revision"] == config["model_revision"]
    scores = {row["pair_id"]: row for row in json.loads(archived["exports/pair_scores.json"])}
    assert len(scores) == len(spec["cases"]) == 7
    for case in spec["cases"]:
        case.pop("swap")
        for side, branch in case["branches"].items():
            branch.pop("archived_full")
            branch["archived_native"] = {
                order: {key: row["original"][key] for key in (
                    "logits", "prediction", "prompt_sha256", "visual_tokens", "lm_tokens"
                )}
                for order, row in scores[case["pair_id"]]["branches"][side]["answers"].items()
            }
            assert len(branch["archived_native"]) == 2
    sources.update({
        "provenance/droid_attention_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_attention_runner.py": (
            ROOT / "scripts/autodl_droid_terminal_attention.py"
        ).read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_state_swap", "hermes_droid_terminal_attention")
        .replace("HERMES_DROID_STATE_SWAP_VALIDATE_ONLY", "HERMES_DROID_TERMINAL_ATTENTION_VALIDATE_ONLY")
        .replace("provenance/droid_swap_runner.py", "provenance/droid_attention_runner.py")
        .replace("7 DROID pairs, 2 fixed-length earlier-state conditions", "7 DROID pairs, 4 attention conditions"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Replay the full history and compare terminal attention\n")
    put(8, get(8).replace("HERMES_droid_state_swap_evidence.zip", "HERMES_droid_terminal_attention_evidence.zip")
        .replace("'droid_state_swap.log'", "'droid_terminal_attention.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-terminal-attention-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
