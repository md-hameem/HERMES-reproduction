"""Check the visual review archive and join the saved analyst observations."""
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from zipfile import ZipFile

from make_confirmation_visual_review_notebook import DEST, ROOT, reference_data
from validate_causal_pilot_evidence import sha, write_csv

DATE = "2026-11-21"
ARCHIVE = ROOT / "evidence/archives/HERMES_confirmation_visual_review.zip"
NOTEBOOK = ROOT / f"notebooks/executed/{DATE}_hermes-confirmation-visual-review.ipynb"
ARCHIVE_SHA = "1c3facd389dda4d2d92d6064ee193da44422471e37fb08cb6d31f474504bb1d9"
NOTEBOOK_SHA = "003fc0fcf0f78dee77137ed65b53a597784e1b26c330c0f4065275b62827faf9"
ORIGINALS = (
    ("sample_112_q3", 273, (1920, 1080), "89ff1f9ac9154bb7889b18bfc6c2153b54cf94fa19912c268129c2223346c55e"),
    ("sample_121_q3", 278, (1920, 960), "629938fd57e6e715fc655d6add83ff6d8c4e50c4baafc208cb06e542ec168919"),
)


def verify_originals_and_sensitivity(tables, cases):
    from PIL import Image
    from validate_short_query_confirmation_evidence import ARCHIVE as PARENT_ARCHIVE

    originals = []
    for case_id, index, size, digest in ORIGINALS:
        path = tables / "originals" / f"{case_id}_sample_{index:04d}.png"
        assert sha(path.read_bytes()) == digest
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size == size
            image.verify()
        frame = next(f for f in cases[case_id]["frames"] if f["sampled_index"] == index)
        originals.append(dict(case_id=case_id, filename=path.name, sha256=digest, **frame))
    by_video, correct = defaultdict(list), Counter()
    with ZipFile(PARENT_ARCHIVE) as zipped:
        for name in zipped.namelist():
            if not name.startswith("checkpoints/") or name.endswith("/sample_121_q3.json"):
                continue
            compact, latest, all_history = json.loads(zipped.read(name))["rows"]
            correct.update({r["method"]: r["correct"] for r in (compact, latest, all_history)})
            by_video[compact["sample_id"]].append(all_history["correct"] - compact["correct"])
    rng = random.Random(2024)
    groups = [(sum(v), len(v)) for v in by_video.values()]
    assert len(groups) == 30 and sum(n for _, n in groups) == 149
    draws = []
    for _ in range(10000):
        draw = rng.choices(groups, k=len(groups))
        draws.append(100 * sum(w for w, _ in draw) / sum(n for _, n in draw))
    draws.sort()
    return dict(originals=originals, exploratory_exclusion=dict(case_id="sample_121_q3", questions=149,
        correct=dict(correct), all_history_vs_compact_pp=100 * sum(w for w, _ in groups) / 149,
        ci_low_pp=draws[250], ci_high_pp=draws[9750]))


def audit(tables, destination):
    from PIL import Image

    assert sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA and sha(NOTEBOOK.read_bytes()) == NOTEBOOK_SHA
    expected = reference_data()
    cases = {c["case_id"]: c for c in expected["cases"]}
    outcomes = {r["case_id"]: r for r in expected["outcomes"]}
    notebook = json.loads(NOTEBOOK.read_bytes())
    template = json.loads(DEST.read_bytes())
    sources = {c["id"]: c["source"] for c in notebook["cells"]}
    assert all(sources[c["id"]] == c["source"] for c in template["cells"])
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    assert not any(o["output_type"] == "error" for o in outputs)
    streams = "\n".join("".join(o["text"]) for o in outputs if o["output_type"] == "stream")
    assert f"SHA256={ARCHIVE_SHA}" in streams
    assert all(f"VISUAL_CASE_READY {key}" in streams for key in cases)
    with ZipFile(ARCHIVE) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names)) and zipped.testzip() is None
        for item in zipped.infolist():
            path = PurePosixPath(item.filename)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
            assert ((item.external_attr >> 16) & 0o170000) != 0o120000

        def read_csv(name):
            return list(csv.DictReader(io.StringIO(zipped.read(name).decode("utf-8"))))

        review_ref = json.loads(zipped.read("review_reference.json"))
        assert review_ref["cases"] == expected["cases"] and review_ref["archive_sha256"] == expected["archive_sha256"]
        assert read_csv("after_review/outcome_key.csv") == expected["outcomes"]
        input_notes = read_csv("review_notes.csv")
        assert [r["case_id"] for r in input_notes] == list(cases)
        assert all(not v for r in input_notes for k, v in r.items() if k != "case_id")
        frame_slots = [dict(case_id=c["case_id"], **f) for c in cases.values() for f in c["frames"]]
        slots = read_csv("frame_slots.csv")
        assert slots == [{k: str(v) for k, v in r.items()} for r in frame_slots]

        expected_sheets = {}
        for key, case in cases.items():
            expected_sheets[f"sheets/{key}/recent.jpg"] = (1150, 752)
            older = sum(f["region"] == "older" for f in case["frames"])
            for start in range(0, older, 8):
                count = min(8, older-start)
                expected_sheets[f"sheets/{key}/older_{start//8+1}.jpg"] = (1230, 48 + 203*((count+3)//4))
        assert set(names) == set(expected_sheets) | {
            "index.html", "review_notes.csv", "frame_slots.csv", "review_reference.json", "after_review/outcome_key.csv"}
        for n, size in expected_sheets.items():
            with Image.open(io.BytesIO(zipped.read(n))) as image:
                assert image.format == "JPEG" and image.size == size, (n, image.size, size)
                image.verify()
        index = zipped.read("index.html").decode()
        assert all(f'src="{n}"' in index for n in expected_sheets)
        assert "correct_arm" not in index and "gold_choice" not in index

    with (tables / "analyst_observations.csv").open(encoding="utf-8", newline="") as handle:
        observations = list(csv.DictReader(handle))
    assert [r["case_id"] for r in observations] == list(cases)
    support_counts = Counter()
    joined = []
    for obs in observations:
        key = obs["case_id"]
        case = cases[key]
        golden = outcomes[key]
        frames = {f["sampled_index"]: f for f in case["frames"]}
        indices = json.loads(obs["supporting_sample_indices"])
        assert len(indices) == len(set(indices)) and all(i in frames for i in indices)
        asserted = obs["support_region"]
        assert asserted in review_ref["support_regions"]
        required = {"recent_only": {"recent"}, "older_only": {"older"},
                    "both_independently": {"recent", "older"}, "needs_both": {"recent", "older"}}
        if asserted in required:
            assert {frames[i]["region"] for i in indices} == required[asserted]
        support_counts[asserted, golden["correct_arm"]] += 1
        joined.append(dict(**obs, query_time_s=case["query_time_s"],
                           supporting_times_s=[frames[i]["start_s"] for i in indices],
                           **{k: v for k, v in golden.items() if k != "case_id"}))
    counts = [dict(support_region=region, all_history_wins=support_counts[region, "all_history"],
                   compact_wins=support_counts[region, "compact_recent4"])
              for region in review_ref["support_regions"]]
    write_csv(tables/"observations_with_outcomes.csv", joined)
    write_csv(tables/"support_summary.csv", counts)
    result = dict(status="pass", archive_sha256=ARCHIVE_SHA, notebook_sha256=NOTEBOOK_SHA,
                  parent_archive_sha256=expected["archive_sha256"], clips=10, cases=12,
                  sheets=len(expected_sheets), frame_slots=len(frame_slots), uploaded_notes_filled=0,
                  original_resolution_frames=0, analyst_support_counts=counts,
                  verification_limit="Saved frame maps and JPEGs verified. Raw video decoding was not repeated locally.")
    result["original_frame_supplement"] = verify_originals_and_sensitivity(tables, cases)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / f"results/confirmation_visual_review_{DATE}",
                           ROOT / f"evidence/audits/confirmation_visual_review_{DATE}_audit.json"), indent=2))
