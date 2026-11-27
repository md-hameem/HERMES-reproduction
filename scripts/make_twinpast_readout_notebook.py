"""Embed the audited fixed-frame readout experiment in a standalone AutoDL notebook."""
import ast
import json

from validate_twinpast_refinement_evidence import ARCHIVE, MANIFEST, NOTEBOOK, ROOT, audit, digest, verify_inventory

TEMPLATE = ROOT / "notebooks/autodl/hermes-twinpast-refinement.ipynb"
DEST = ROOT / "notebooks/autodl/hermes-twinpast-fixed-frame-readout.ipynb"

INTRO = """# TwinPast fixed-frame readout

Use the **same AutoDL container** as the refinement run if possible. Open this
notebook and select **Restart Kernel → Run All**. The notebook probes and
reuses the refinement environment, verified 7B cache and 16 existing clips.
A fresh container may need around 40 GiB free and must download the assets.

Replay the archived direct readout on both the coarse uniform and RGB-change
frame sets. A replay error stops the comparison and preserves its discrepancy
report. On a pass, generate one short observation per branch and frame set
without showing any answer options. Score both option orders with the same
frames and observation. Report coarse uniform as the primary comparison and
RGB-change as a separate diagnostic. The run counts the observation's extra
decoder calls, input tokens and measured forward time.

The eight pairs are development footage. Advance only if the primary
comparison recovers a robust pair without losing its robust baseline pair.

After **TWINPAST_READOUT_COMPLETE**, save the executed notebook and download
**HERMES_twinpast_readout_evidence.zip**. If the replay fails, run the download
cell manually and send the failure evidence.
"""


def archive_cases(saved):
    previous = json.loads(saved["provenance/refinement_cases.json"])
    prior_cases = {row["pair_id"]: row for row in previous["cases"]}
    media = {plan["pair_id"]: {side: dict(file=plan[f"video_{side}"], bytes=r["bytes"], sha256=r["sha256"])
                              for side, r in receipt["decoded"].items()}
             for plan, receipt, _ in previous["media_plan"]}
    accepted = json.loads(saved["exports/run_config.json"])["accepted"]
    cases = []
    for pair in accepted:
        block = json.loads(saved[f"checkpoints/{pair}.json"])["block"]
        branches = {}
        for side, branch in block["branches"].items():
            coarse, fine = branch["coarse"], branch["refined"]
            choice_calls = {(r["phase"], r["order"]): r for r in branch["calls"] if r["phase"] != "shared_retrieval"}
            rgb_slots = branch["refined_picks"]["state_change"]
            direct = {}
            for method in ("coarse_uniform", "state_change"):
                direct[method] = {}
                for order, answers in branch["answers"].items():
                    call = choice_calls[method, order]
                    direct[method][order] = {**answers[method], **{k: call[k] for k in
                                                                  ("lm_tokens", "visual_tokens", "prompt_sha256")}}
            assert sorted(rgb_slots) == rgb_slots and len(rgb_slots) == 6
            branches[side] = dict(coarse=dict(indices=coarse["indices"], hashes=coarse["hashes"],
                                             slots=branch["coarse_uniform_slots"]),
                                  rgb={k: [fine[k][i] for i in rgb_slots] for k in ("indices", "hashes")},
                                  direct=direct)
        assert branches["a"]["coarse"]["hashes"][-4:] == branches["b"]["coarse"]["hashes"][-4:]
        cases.append(dict(**prior_cases[pair], branches=branches, media=media[pair]))
    return dict(cases=cases, model_revision=previous["model_revision"], video_processor=previous["video_processor"],
                media_source=previous["media_source"], source_archive_sha256=digest(ARCHIVE.read_bytes()))


def notebook():
    assert audit(ARCHIVE, NOTEBOOK)["pairs"] == 8
    saved = verify_inventory(ARCHIVE)
    spec = archive_cases(saved)
    original_targets = json.loads(saved["provenance/evaluation_targets.json"])
    targets = {row["pair_id"]: original_targets[row["pair_id"]] for row in spec["cases"]}
    book = json.loads(TEMPLATE.read_bytes())
    cells = book["cells"]

    def get(i):
        return "".join(cells[i]["source"])

    def put(i, source):
        cells[i]["source"] = source.splitlines(keepends=True)

    base_sources = ast.literal_eval(ast.parse(get(4)).body[0].value)
    assert all(src.encode() == saved[name] for name, src in base_sources.items())
    sources = {name: src for name, src in base_sources.items() if name not in
               {"provenance/refinement_cases.json", "provenance/refinement_runner.py"}}
    sources.update({
        "provenance/evaluation_targets.json": json.dumps(targets, indent=2) + "\n",
        "provenance/readout_cases.json": json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "provenance/readout_runner.py": (ROOT / "scripts/autodl_twinpast_readout.py").read_text(encoding="utf-8"),
    })
    hashes = {name: digest(src.encode()) for name, src in sources.items()}
    put(0, INTRO)
    put(1, get(1).replace("/hermes_twinpast_refinement", "/hermes_twinpast_readout")
        .replace("REUSE = Path('/root/autodl-tmp/hermes_causal_pilot')",
                 "REUSE = WORK.with_name('hermes_twinpast_refinement')\nFALLBACK = WORK.with_name('hermes_causal_pilot')")
        .replace("HERMES_TWINPAST_REFINEMENT_VALIDATE_ONLY", "HERMES_TWINPAST_READOUT_VALIDATE_ONLY")
        .replace("provenance/refinement_runner.py", "provenance/readout_runner.py")
        .replace("Up to 8 fresh pairs, 3 matched methods", "8 archived development pairs, 2 fixed frame sets")
        .replace("MEDIA_REUSE = Path('/root/autodl-tmp/twinpast-development')\n", "")
        .replace("PREVIOUS = WORK.with_name('hermes_twinpast_temporal_windows')", "PREVIOUS = REUSE"))
    launch_checks = get(3).replace(
        "    env.update(HF_HOME=str((REUSE if (REUSE/'cache/huggingface').is_dir() else WORK)/'cache/huggingface'),",
        "    model_folder = 'models--' + MODEL_ID.replace('/', '--')\n"
        "    candidates = [root/'cache/huggingface' for root in (REUSE, FALLBACK, WORK)]\n"
        "    snapshots = [root/'hub'/model_folder/'snapshots'/MODEL_SHA for root in candidates]\n"
        "    cache = next((root for root, snap in zip(candidates, snapshots, strict=True)\n"
        "                  if sum(p.is_file() for p in snap.glob('model-*-of-00004.safetensors')) == 4), None)\n"
        "    if cache is None:\n"
        "        cache = next((root for root, snap in zip(candidates, snapshots, strict=True) if snap.is_dir()),\n"
        "                     candidates[-1])\n"
        "    env.update(HF_HOME=str(cache),",
    ).replace(
        "    minimum_free = 1 if (REUSE/'venv/bin/python').is_file() and (REUSE/'cache/huggingface').is_dir() else 18",
        "    minimum_free = 1 if any((root/'venv/bin/python').is_file() for root in (REUSE, FALLBACK)) else 18",
    )
    put(3, launch_checks)
    put(4, f"SOURCES = {sources!r}\nEXPECTED = {hashes!r}\n" + "\n".join(get(4).split("\n")[2:]))
    setup = get(5)
    lo = setup.index("    candidate_python")
    hi = setup.index("    PIP_ENV", lo)
    setup = setup[:lo] + (
        "    for root in (REUSE, FALLBACK):\n"
        "        candidate_python = root/'venv/bin/python'\n"
        "        if candidate_python.is_file() and probe([candidate_python, '-c', ENV_PROBE], ENV).returncode == 0:\n"
        "            PYTHON = candidate_python\n"
        "            print('REUSING_VERIFIED_PINNED_ENVIRONMENT', PYTHON, flush=True)\n"
        "            break\n"
    ) + setup[hi:]
    torch_install = next(line for line in setup.splitlines() if "install(['torch @ https://" in line)
    setup = setup.replace(torch_install, """        wheel_name = 'torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
        wheel = next((p for root in (REUSE, FALLBACK, WORK)
                      if (p := root/'downloads'/wheel_name).is_file() and zipfile.is_zipfile(p)), None)
        if wheel is None:
            wheel = WORK/'downloads'/wheel_name
            wheel.parent.mkdir(parents=True, exist_ok=True)
            partial = wheel.with_suffix('.whl.part')
            if not zipfile.is_zipfile(partial):
                url = ('https://download-r2.pytorch.org/whl/cu128/'
                       'torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl')
                logged(['curl', '-fL', '-C', '-', '--connect-timeout', '20', '--speed-limit', '16384',
                        '--speed-time', '90', '--progress-bar', '-o', partial, url], 'torch_download.log', ENV)
            if not zipfile.is_zipfile(partial):
                raise ValueError(f'Incomplete PyTorch wheel: {partial}')
            partial.replace(wheel)
        install([str(wheel)])""")
    put(5, setup)
    put(7, "## 2. Replay the archived scores and compare the readouts\n")
    put(8, get(8).replace("HERMES_twinpast_refinement_evidence.zip", "HERMES_twinpast_readout_evidence.zip")
        .replace("'--media-reuse', MEDIA_REUSE, ", "").replace("'twinpast_refinement.log'", "'twinpast_readout.log'"))
    for i, cell in enumerate(cells):
        cell["id"] = f"twinpast-readout-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return book


if __name__ == "__main__":
    DEST.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST}")
