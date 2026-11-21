"""Build the CPU-only visual review notebook from the audited archive."""
import hashlib
import json
from zipfile import ZipFile

from validate_causal_pilot_evidence import check_manifest
from validate_short_query_confirmation_evidence import ARCHIVE, ARCHIVE_SHA, ARMS, ROOT

DEST = ROOT / "notebooks/autodl/hermes-confirmation-visual-review.ipynb"
INTRO = """# HERMES visual evidence review

Upload this notebook to the **same AutoDL container** and choose **Restart Kernel → Run All**.
Use a CPU session if available. It reads the existing 10 clips under
`/root/autodl-tmp/hermes_short_query_confirmation_v2/videos`. No downloads or
model inference are needed.

It renders the exact pre-query frame slots for the **12 questions** where exactly one
of the primary methods was correct. The recent sheet shows the four frames that
compact recent-four received. The older sheets show additional frames used during
all-history encoding. The frame IDs and timestamps come from the saved run, and the
runner checks each clip hash before decoding.

Review the sheets **before** consulting the separate outcome key. Record where
decisive evidence is visible and which sample indices support your judgement.
Use **uncertain** for ambiguous or motion-dependent cases. Thumbnails can hide
small text. Use the original-frame helper to check it before assigning
**not_visible**. This is a diagnostic review of disagreements, so it cannot
estimate how common a mechanism is in the benchmark.
"""


def reference_data():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
    with ZipFile(ARCHIVE) as zipped:
        manifest, payload = check_manifest(zipped)
    assert manifest["status"] == "complete"

    def read(n):
        return json.loads(payload[n])

    videos = {v["sample_id"]: v["sha256"] for v in read("asset_manifest.json")["videos"]}
    timing = {sid: read(f"provenance/features_{sid}.json")["timestamps"] for sid in videos}
    cases, outcomes = [], []
    for name in sorted(n for n in payload if n.startswith("checkpoints/")):
        rows = {r["method"]: r for r in read(name)["rows"]}
        a, c = rows[ARMS[2]], rows[ARMS[0]]
        if a["correct"] == c["correct"]:
            continue
        case_id = name.rsplit("/", 1)[-1].removesuffix(".json")
        sid = a["sample_id"]
        prep = read(f"preparation/{case_id}.json")
        history, recent = prep["history_indices"], prep["recent_indices"]
        assert a["recent_indices"] == c["recent_indices"] == recent and a["selected_history_indices"] == history
        assert a["choices"] == c["choices"] and a["gold_choice"] == c["gold_choice"]
        frames = []
        for index in history + recent:
            saved = timing[sid][index]
            assert int(saved["sampled_index"]) == index and float(saved["start_s"]) <= a["query_time_s"] + 1e-6
            frames.append(dict(sampled_index=index, source_frame=int(saved["source_frame"]),
                               start_s=float(saved["start_s"]), region="recent" if index in recent else "older"))
        cases.append(dict(case_id=case_id, sample_id=sid, video_sha256=videos[sid],
                          query_time_s=a["query_time_s"], question=a["question"], choices=a["choices"], frames=frames))
        outcomes.append(dict(case_id=case_id, gold_choice=a["gold_choice"], gold_answer=a["gold_answer"],
                             compact_choice=c["pred_choice"], all_history_choice=a["pred_choice"],
                             correct_arm="all_history" if a["correct"] else "compact_recent4"))
    assert len(cases) == 12 and len({c["sample_id"] for c in cases}) == 10
    assert sum(r["correct_arm"] == "all_history" for r in outcomes) == 4
    return dict(archive_sha256=ARCHIVE_SHA, cases=cases, outcomes=outcomes)


CONFIG = """from pathlib import Path
import csv, hashlib, json, os, shutil, subprocess

BASE = Path('/root/autodl-tmp')
MEDIA = BASE / 'hermes_short_query_confirmation_v2/videos'
PYTHON = BASE / 'hermes_causal_pilot/venv/bin/python'
WORK = BASE / 'hermes_confirmation_visual_review'
VALIDATE_ONLY = os.environ.get('HERMES_VISUAL_REVIEW_VALIDATE_ONLY') == '1'
if not VALIDATE_ONLY:
    from IPython.display import FileLink, Image, Markdown, display
"""

SETUP = """assert hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest() == RUNNER_SHA
assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == REFERENCE_SHA
reference = json.loads(REFERENCE_JSON)
cases = {c['case_id']: c for c in reference['cases']}
if not VALIDATE_ONLY:
    if not PYTHON.is_file():
        raise FileNotFoundError(f'Shared experiment Python is missing: {PYTHON}')
    WORK.mkdir(parents=True, exist_ok=True)
    runner, ref = WORK/'visual_review.py', WORK/'reference.json'
    for path, value in ((runner, RUNNER_SOURCE), (ref, REFERENCE_JSON)):
        if path.exists() and path.read_text(encoding='utf-8') != value:
            raise RuntimeError('Review sources changed. Use a different WORK to preserve earlier notes.')
        path.write_text(value, encoding='utf-8')
    command = [str(PYTHON), str(runner), '--reference', str(ref), '--media', str(MEDIA), '--output', str(WORK)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end='', flush=True)
        if proc.wait():
            raise RuntimeError('Visual review preparation failed. See the output above.')
    print(f'Ready: {len(cases)} cases. Review notes: {WORK / "review_notes.csv"}')
else:
    print('VISUAL_REVIEW_VALIDATION_PASS')
"""

REVIEW = r"""def show_case(case_id):
    case = cases[case_id]
    display(Markdown(f"**{case_id} · {case['query_time_s']:.2f} s**\n\n"
                     + case['question'] + '\n\n'
                     + '\n'.join(f'- {chr(65+i)}. {v}' for i, v in enumerate(case['choices']))))
    folder = WORK / 'sheets' / case_id
    for path in [folder/'recent.jpg', *sorted(folder.glob('older_*.jpg'))]:
        display(Image(filename=str(path)))

def show_original(case_id, sampled_index):
    result = subprocess.run(command + ['--inspect', case_id, str(sampled_index)],
                            check=True, capture_output=True, text=True)
    display(Image(filename=result.stdout.strip().splitlines()[-1]))

if not VALIDATE_ONLY:
    print('Case IDs:', ', '.join(cases))
    show_case(next(iter(cases)))
"""

RECORD = """def record_review(case_id, reviewer, support_region, sample_ids=(), choice='', confidence='', notes='', ambiguity_or_motion=''):
    case = cases[case_id]
    sample_ids = tuple(sample_ids)
    permitted = {'recent_only', 'older_only', 'both_independently', 'needs_both', 'not_visible', 'uncertain'}
    frames = {f['sampled_index']: f['region'] for f in case['frames']}
    if support_region not in permitted or not reviewer.strip():
        raise ValueError('Choose a listed support region and enter a reviewer label.')
    if any(i not in frames for i in sample_ids):
        raise ValueError('Use the sample indices shown on this case.')
    required = {'recent_only': {'recent'}, 'older_only': {'older'},
                'both_independently': {'recent', 'older'}, 'needs_both': {'recent', 'older'}}
    if support_region in required and {frames[i] for i in sample_ids} != required[support_region]:
        raise ValueError('Support indices do not match the stated region.')
    if choice and choice not in [chr(65+i) for i in range(len(case['choices']))]:
        raise ValueError('Use an option letter from this question.')
    if confidence not in ('', 'low', 'medium', 'high'):
        raise ValueError('Use low, medium or high.')
    path = WORK/'review_notes.csv'
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    row = next(r for r in rows if r['case_id'] == case_id)
    row.update(reviewer=reviewer, support_region=support_region,
               supporting_sample_indices=json.dumps(list(sample_ids)), reviewer_choice=choice,
               confidence=confidence, ambiguity_or_motion=ambiguity_or_motion, notes=notes)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    print(f'Saved {case_id}')

# Use a new cell for each case. Example shape with placeholders:
# record_review(case_id, reviewer, support_region, sample_ids, choice, confidence, notes)
"""

EXPORT = """if not VALIDATE_ONLY:
    archive = WORK/'HERMES_confirmation_visual_review.zip'
    subprocess.run(command + ['--zip', str(archive)], check=True)
    link = Path.cwd()/archive.name
    if not link.is_symlink() and link.resolve() != archive.resolve():
        shutil.copy2(archive, link)
    if link.is_symlink():
        raise ValueError('Download copy path is a symlink. Use the ZIP in WORK.')
    display(FileLink(link.name))
"""


def notebook():
    cells = []

    def cell(kind, text):
        item = dict(cell_type=kind, id=f"visual-review-{len(cells):02d}", metadata={}, source=text.splitlines(keepends=True))
        if kind == "code":
            compile(text, item["id"], "exec")
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", INTRO)
    cell("code", CONFIG)
    cell("markdown", "## 1. Render the frozen frames\nThe embedded data pins the clip hashes and input frame slots.\n")
    source = (ROOT/"scripts/autodl_confirmation_visual_review.py").read_text(encoding="utf-8")
    ref = json.dumps(reference_data(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    header = "\n".join([f"RUNNER_SOURCE = {source!r}", f"REFERENCE_JSON = {ref!r}",
                        f"RUNNER_SHA = {hashlib.sha256(source.encode()).hexdigest()!r}",
                        f"REFERENCE_SHA = {hashlib.sha256(ref.encode()).hexdigest()!r}", ""])
    cell("code", header + SETUP)
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    cell("markdown", "## 2. Examine each question\nCall `show_case(case_id)`. For detail, call `show_original(case_id, sampled_index)` using a displayed sample number.\n")
    cell("code", REVIEW)
    cell("markdown", """## 3. Save your observations

Use `record_review` in a new cell or edit `review_notes.csv` directly.
Choose `recent_only`, `older_only`, `both_independently`, `needs_both`,
`not_visible` or `uncertain`. If both windows have support, distinguish
either window being sufficient from needing evidence across both.
Record the sample indices, your best option if you can tell, and any ambiguity
or motion that the sampled stills miss. Leave the separate outcome key for later.
""")
    cell("code", RECORD)
    cell("markdown", "## 4. Download the review package\nRerun this cell after adding notes. Open the extracted `index.html` for all cases. The outcome key is under `after_review/`.\n")
    cell("code", EXPORT)
    return dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3 (ipykernel)", language="python", name="python3"),
        language_info=dict(name="python", version="3.12.0")), nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    value = notebook()
    DEST.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {DEST} ({len(value['cells'])} cells)")
