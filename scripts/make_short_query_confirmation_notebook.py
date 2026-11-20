"""Embed the frozen clip manifest and the three-arm AutoDL runner."""
import hashlib
import json

import make_construction_selection_notebook as selection
from make_confirmation_manifest import OUTPUT, build

ROOT = selection.ROOT
DEST = ROOT / "notebooks/autodl/hermes-short-query-confirmation.ipynb"
INTRO = """# HERMES short query-cache confirmation

Open this notebook in the **same AutoDL container**, then choose **Restart Kernel → Run All**.
Defaults reuse the pinned environment and model. The run writes to
`/root/autodl-tmp/hermes_short_query_confirmation`.

**30 frozen StreamingBench clips, 150 questions, 450 arm-level answers.** This is a
clip-disjoint test within the same benchmark. Original source overlap is unverified.
The three fixed arms use identical answer-cache sizes:

- Compact recent-four, with no older encoding inputs.
- Latest-two contextual, using the nearest two available older frames during encoding.
- All-history contextual, using up to 16 older frames during encoding.

The contextual arms preserve original recent-token positions and use no older slots
during answering. All-history versus compact recent-four is the primary accuracy
comparison. The evidence includes paired wins and losses, a 95% video bootstrap
interval, and leave-one-video sensitivity. A 30-clip result may remain inconclusive.

Allow space and time to download approximately **4.85 GB of selected video members**
plus range overhead and generated features. The notebook reads only the frozen clips
and will stop if range access fails on both endpoints. Completed media and QA
checkpoints are reusable. New features come from a causal 0.5 FPS grid and only
the windows needed by the frozen questions. No additional images or annotations
are needed. The two-hour soft budget starts after model load and is checked between
videos. Downloads are separate.

When you see `SHORT_QUERY_CONFIRMATION_COMPLETE`, download
**HERMES_short_query_confirmation_evidence.zip** and save the executed notebook.
"""


def notebook():
    nb = selection.notebook()
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        cell["id"] = f"confirmation-{i:02d}"

    def put(i, content):
        if cells[i]["cell_type"] == "code":
            compile(content, cells[i]["id"], "exec")
        cells[i]["source"] = content.splitlines(keepends=True)

    put(0, INTRO)
    config = "".join(cells[1]["source"]).replace("hermes_construction_selection", "hermes_short_query_confirmation").replace(
        "HERMES_SELECTION_VALIDATE_ONLY", "HERMES_CONFIRMATION_VALIDATE_ONLY").replace(
        "240 construction-selection answers", "450 answers on 30 frozen clips").replace("MAX_HOURS = 1.0", "MAX_HOURS = 2.0")
    put(1, config)
    manifest = json.loads(OUTPUT.read_bytes())
    if manifest != build():
        raise ValueError("Manifest differs from the committed selection rule")
    sources = dict(selection.short.matched.SOURCES, RUNNER_SOURCE="autodl_short_query_confirmation.py",
                   SHORT_BASE_SOURCE="autodl_short_query.py", MATCHED_BASE_SOURCE="autodl_matched_boundary.py",
                   SELECTION_BASE_SOURCE="autodl_construction_selection.py")
    parts = []
    for variable, name in sources.items():
        code = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "'''" not in code
        digest = hashlib.sha256(code.encode()).hexdigest()
        parts.extend([f"{variable} = r'''" + code + "'''\n", f"ast.parse({variable})\n",
                      f"assert hashlib.sha256({variable}.encode()).hexdigest() == {digest!r}\n"])
    ref = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts.append(f"REFERENCE_JSON = {ref!r}\nassert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(ref.encode()).hexdigest()!r}\n")
    embed = selection.short.matched.stage.EMBED.replace("('stage_runner.py',RUNNER_SOURCE),",
        "('stage_runner.py',RUNNER_SOURCE),('provenance/stage_base.py',STAGE_BASE_SOURCE),"
        "('provenance/matched_base.py',MATCHED_BASE_SOURCE),('provenance/short_base.py',SHORT_BASE_SOURCE),"
        "('provenance/selection_base.py',SELECTION_BASE_SOURCE),").replace(
        "            target=WORK/name\n", "            target=WORK/name\n            target.parent.mkdir(parents=True,exist_ok=True)\n")
    put(8, "".join(parts) + embed)
    put(9, "## 3. Acquire frozen clips and run the three-arm comparison\n")
    put(10, "".join(cells[10]["source"]).replace("'selection.log'", "'confirmation.log'"))
    put(12, "".join(cells[12]["source"]).replace(
        selection.ARTIFACT, "HERMES_short_query_confirmation_evidence.zip"))
    return nb


if __name__ == "__main__":
    nb = notebook()
    DEST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(nb['cells'])} cells)")
