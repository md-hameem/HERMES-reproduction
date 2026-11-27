"""Build the matched-length DROID early-state swap notebook."""
import ast
import json

from prepare_droid_state_swap import DEST as PLAN
from validate_droid_terminal_evidence import ARCHIVE, MANIFEST, ROOT, TEMPLATE, digest, verify_inventory

DEST = ROOT / "notebooks/autodl/hermes-droid-state-swap.ipynb"
INTRO = """# DROID fixed-length earlier-state swap

Use the same AutoDL GPU as the terminal-context run, if available, with at
least 22 GiB of VRAM. Upload this notebook beside the existing
**DROID_natural_history_frames.zip** and select **Restart Kernel → Run All**.

In both directions of each pair, replace the first two history frames with the
other branch's first two frames. The substituted frames show the same object
in the state that agrees with the recipient branch's terminal state. The last
eight positions, including all terminal and recent frames, are identical.

Replay the original 28 ten-frame scores, then report 56 matched choice scores,
robust pair gains and losses, and changes in the gold-versus-alternative logit
margin. The input length and slots are fixed. The cross-episode splice can also
change robot pose, lighting and other image details, so this is an exploratory
content intervention rather than a state-only manipulation.

Frames are resized from [DROID](https://droid-dataset.github.io/), Khazatsky et al.
and the DROID Dataset Team, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

After **DROID_STATE_SWAP_COMPLETE**, save the executed notebook and download
**HERMES_droid_state_swap_evidence.zip**. If the notebook stops after launch,
run the final download cell manually to retrieve the failure evidence.
"""


def notebook():
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    archived = verify_inventory(ARCHIVE)
    swap = json.loads(PLAN.read_bytes())
    assert swap["terminal_context_archive_sha256"] == digest(ARCHIVE.read_bytes())
    assert swap["natural_history_manifest_sha256"] == digest(MANIFEST.read_bytes())
    sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == archived[n] for n, src in sources.items())
    spec = json.loads(sources.pop("provenance/droid_terminal_cases.json"))
    sources.pop("provenance/droid_terminal_runner.py")
    spec.pop("archived_full_zip_sha256")
    spec.pop("archived_full_gpu")
    assert spec["frame_zip_sha256"] == swap["frame_zip_sha256"]
    plan_cases = {r["pair_id"]: r for r in swap["cases"]}
    scores = {r["pair_id"]: r for r in json.loads(archived["exports/pair_scores.json"])}
    assert len(plan_cases) == len(scores) == len(spec["cases"]) == 7
    for case in spec["cases"]:
        pair = case["pair_id"]
        case["swap"] = plan_cases[pair]
        assert case["recent_frames"] == case["swap"]["shared_recent_frames"]
        assert case["swap"]["replaced_history_slots"] == [0, 1]
        for side, branch in case["branches"].items():
            branch["archived_full"] = {
                order: {k: row["full_annotated"][k]
                        for k in ("logits", "prediction", "prompt_sha256", "visual_tokens", "lm_tokens")}
                for order, row in scores[pair]["branches"][side]["answers"].items()
            }
    spec.update(prior_archive_sha256=swap["terminal_context_archive_sha256"],
                prior_gpu=json.loads(archived["exports/run_config.json"])["gpu"],
                swap_plan_sha256=digest(PLAN.read_bytes()))
    sources.update({
        "provenance/droid_swap_plan.json": PLAN.read_text(encoding="utf-8"),
        "provenance/droid_swap_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/droid_swap_runner.py": (ROOT / "scripts/autodl_droid_state_swap.py").read_text(encoding="utf-8"),
    })
    hashes = {n: digest(src.encode()) for n, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("hermes_droid_terminal_context", "hermes_droid_state_swap")
        .replace("HERMES_DROID_TERMINAL_CONTEXT_VALIDATE_ONLY", "HERMES_DROID_STATE_SWAP_VALIDATE_ONLY")
        .replace("provenance/droid_terminal_runner.py", "provenance/droid_swap_runner.py")
        .replace("7 DROID pairs, 3 context lengths", "7 DROID pairs, 2 fixed-length earlier-state conditions"))
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    put(7, "## 2. Replay the full history and compare the symmetric swap\n")
    put(8, get(8).replace("HERMES_droid_terminal_context_evidence.zip", "HERMES_droid_state_swap_evidence.zip")
        .replace("'droid_terminal_context.log'", "'droid_state_swap.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"droid-swap-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
