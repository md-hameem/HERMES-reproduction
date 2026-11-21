"""Generate a self-contained AutoDL notebook for the frozen target-binding study."""
import hashlib
import json

import make_short_query_confirmation_notebook as confirmation
from make_target_binding_manifest import OUTPUT, ROOT, build

DEST = ROOT / "notebooks/autodl/hermes-target-binding.ipynb"
ARTIFACT = "HERMES_target_binding_evidence.zip"
REUSE_CHECKOUT = """if not VALIDATE_ONLY:
    candidate_repo = REUSE/'repository'
    if (candidate_repo/'.git').is_dir():
        revision = probe(['git', 'rev-parse', 'HEAD'], ENV, candidate_repo)
        if revision.returncode == 0 and revision.stdout.strip() == BASE_REVISION and all(
            (candidate_repo/rel).is_file() and
            hashlib.sha256((candidate_repo/rel).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest() == expected
            for rel, expected in SOURCE_HASHES.items()
        ):
            REPO = candidate_repo
            ENV['PYTHONDONTWRITEBYTECODE'] = '1'
            print('PINNED_SOURCE_REUSE_PASS', BASE_REVISION, flush=True)
"""
INTRO = """# Target binding for streaming video QA

Open this notebook on the RTX 4090 AutoDL container and choose **Restart Kernel → Run All**.
The pinned environment and model under `/root/autodl-tmp/hermes_causal_pilot` are verified before reuse.
This run has its own work directory, `/root/autodl-tmp/hermes_target_binding`.

**60 frozen clips, 300 questions, 1,800 selector outcomes.** The primary comparison uses the
269 questions with at least two complete four-frame windows. All windows lie within the latest
20 causal samples on the 0.5 FPS grid. One answer per window is reused across selectors that
choose it. No training, extra model, human labels, answer choices, or gold answers are used
to select windows. The original multiple-choice prompt remains the answer prompt.

| Selector | Rule |
|---|---|
| Recent | Newest four frames |
| Previous | Immediately preceding four frames, or recent if unavailable |
| Global relevance | Highest question-to-window relevance margin |
| Target only | Highest target-identification margin |
| Mean | Highest mean of target and target-specific evidence margins |
| Conjunction | Highest minimum of the same two margins |

**Primary contrast** is conjunction versus mean. Both use the same probes and candidates.
The report shows paired video-bootstrap intervals, wins and losses, window switches,
an answer oracle for the fixed candidate set, and a component-time estimate per policy.
The text-rich slice and other baseline contrasts are secondary.

The runner checks selected video sizes from four ZIP indexes before downloading. It requires
space for the selected members plus 4 GiB for working files. It uses HTTP byte ranges and
retains completed downloads and checkpointed questions. Leave `MAX_HOURS = 0.0` for the full
run. To resume, use the same notebook, GPU type and work directory. Transfers depend on
the container connection. Watch for `MEDIA_PLAN`, `VIDEO_READY`, and `VIDEO_BLOCK_DONE`.

At **TARGET_BINDING_COMPLETE**, download the evidence ZIP and save the executed notebook.
"""


def notebook():
    nb = confirmation.notebook()
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        cell["id"] = f"target-binding-{i:02d}"

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, content):
        if cells[i]["cell_type"] == "code":
            compile(content, cells[i]["id"], "exec")
        cells[i]["source"] = content.splitlines(keepends=True)

    put(0, INTRO)
    put(1, get(1).replace("hermes_short_query_confirmation_v2", "hermes_target_binding")
        .replace("HERMES_CONFIRMATION_VALIDATE_ONLY", "HERMES_TARGET_BINDING_VALIDATE_ONLY")
        .replace("MAX_HOURS = 2.0", "MAX_HOURS = 0.0")
        .replace("450 answers on 30 frozen clips", "1,800 selector outcomes on 60 frozen clips"))
    hashes, clone = get(4).split("\n", 1)
    put(4, hashes + "\n" + REUSE_CHECKOUT + clone.replace(
        "if not VALIDATE_ONLY:", "if not VALIDATE_ONLY and REPO == WORK/'repository':", 1))
    manifest = json.loads(OUTPUT.read_bytes())
    if manifest != build():
        raise ValueError("Frozen manifest differs from the metadata selection rule")
    sources = get(8).split("REFERENCE_JSON = ", 1)[0]
    sources = sources.replace("RUNNER_SOURCE = ", "CONFIRMATION_BASE_SOURCE = ", 1)
    sources = sources.replace("ast.parse(RUNNER_SOURCE)", "ast.parse(CONFIRMATION_BASE_SOURCE)", 1)
    sources = sources.replace("sha256(RUNNER_SOURCE.encode())", "sha256(CONFIRMATION_BASE_SOURCE.encode())", 1)
    runner = (ROOT / "scripts/autodl_target_binding.py").read_text(encoding="utf-8")
    if "'''" in runner:
        raise ValueError("Embedded runner contains its raw string delimiter")
    ref = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    embed = get(8)[get(8).index("if not VALIDATE_ONLY:"):]
    embed = embed.replace("('stage_runner.py',RUNNER_SOURCE),",
                          "('stage_runner.py',RUNNER_SOURCE),('provenance/confirmation_base.py',CONFIRMATION_BASE_SOURCE),")
    put(8, sources + "RUNNER_SOURCE = r'''" + runner + "'''\n"
        + f"ast.parse(RUNNER_SOURCE)\nassert hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest() == "
        + f"{hashlib.sha256(runner.encode()).hexdigest()!r}\nREFERENCE_JSON = {ref!r}\n"
        + f"assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref.encode()).hexdigest()!r}\n"
        + embed)
    put(9, "## 3. Acquire the frozen clips and run the six selectors\n")
    put(10, get(10).replace("'confirmation.log'", "'target_binding.log'").replace(
        "if (WORK/'exports/completion.json').is_file():",
        f"if (WORK/'exports/completion.json').is_file() and (WORK/{ARTIFACT!r}).is_file():"))
    put(12, get(12).replace("HERMES_short_query_confirmation_evidence.zip", ARTIFACT))
    return nb


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
