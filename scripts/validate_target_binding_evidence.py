"""Reconstruct target-binding outcomes from immutable per-question evidence."""
import ast
import csv
import io
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import autodl_target_binding as runner
from validate_causal_pilot_evidence import check_manifest, sha

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/archives/HERMES_target_binding_evidence.zip"
AUDIT = ROOT / "evidence/audits/target_binding_2026-11-22_audit.json"
NOTEBOOK = ROOT / "notebooks/executed/2026-11-22_hermes-target-binding.ipynb"


def audit(path, notebook=None):
    with ZipFile(path) as archive:
        inventory, payload = check_manifest(archive)
    assert inventory["status"] == "complete"
    config = json.loads(payload["exports/run_config.json"])
    for name, digest in config["sources"].items():
        assert sha(payload[name]) == digest, name
    launch = json.loads(payload["launch.json"])
    assert launch["runner_sha256"] == config["sources"]["stage_runner.py"]
    archive_sha = sha(path.read_bytes())
    notebook_sha = None
    if notebook is not None:
        notebook_sha = sha(notebook.read_bytes())
        cells = json.loads(notebook.read_bytes())["cells"]
        run_cell, = [c for c in cells if "RUNNER_SOURCE = r" in "".join(c["source"])]
        embedded = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse("".join(run_cell["source"])).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {"REFERENCE_JSON", "RUNNER_SOURCE"}}
        assert embedded["REFERENCE_JSON"].encode() == payload["reference.json"]
        assert embedded["RUNNER_SOURCE"].encode() == payload["stage_runner.py"]
        outputs = "\n".join("".join(o.get("text", "")) for c in cells for o in c.get("outputs", []))
        assert "TARGET_BINDING_COMPLETE" in outputs and archive_sha in outputs
    manifest = json.loads(payload["reference.json"])
    assert manifest == json.loads((ROOT / "data/streamingbench/target_binding_manifest.json").read_bytes())
    assert config["manifest_sha256"] == sha(payload["reference.json"])
    videos = runner.frozen_videos(manifest, ROOT)
    assets = {a["sample_id"]: a for a in json.loads(payload["asset_manifest.json"])["videos"]}
    assert set(assets) == {v["sample_id"] for v in videos}
    for shard in manifest["shards"]:
        receipt = json.loads(payload[f"provenance/media/{shard['shard']}.json"])
        assert receipt["videos"] == [assets[c["sample_id"]] for c in shard["clips"]]
    blocks = []
    for video in videos:
        sid = video["sample_id"]
        features = json.loads(payload[f"provenance/features_{sid}.json"])
        meta = features["metadata"]
        video["timestamps"] = features["timestamps"]
        identity = runner.d.fingerprint(dict(config=config["config_hash"], source=assets[sid]["sha256"]))
        assert meta["identity"] == identity
        for qi in range(len(video["conversations"])):
            value = json.loads(payload[f"checkpoints/sample_{sid}_q{qi}.json"])
            key = runner.d.fingerprint(dict(config=config["config_hash"], features=runner.d.fingerprint(meta),
                                            sample=sid, question=qi))
            assert value["key"] == key
            block = value["block"]
            assert value["block_sha256"] == runner.d.fingerprint(block)
            h, recent = runner.s.split_frames(video, qi)
            expected = [recent] + [h[end-4:end] for end in range(len(h), 3, -4)]
            windows = block["windows"]
            assert [c["indices"] for c in windows] == expected
            assert sorted(block["execution_order"]) == list(range(len(expected)))
            assert len({(c["query_kv_bytes"], c["position_trace"][0]["logical_start"]) for c in windows}) == 1
            for i, c in enumerate(windows):
                assert c["window"] == i
                assert [j for group in c["groups"] for j in group] == c["indices"]
                assert set(c["scores"]) == (set(runner.PROBES) if len(windows) > 1 else set())
                for name, score in c["scores"].items():
                    margins = c["order_margins"][name]
                    assert len(margins) == 2 and math.isclose(score, sum(margins) / 2)
                pred = c["pred_answer"].strip()
                parsed = pred[pred.index(")") - 1:pred.index(")")] if ")" in pred else pred[:1]
                assert c["pred_choice"] == parsed
                assert c["correct"] == int(parsed == runner.d.question_identity(video, qi, "window")["gold_choice"])
            chosen = runner.select([c["scores"] for c in windows])
            assert [row["method"] for row in block["rows"]] == list(runner.ARMS)
            for row in block["rows"]:
                arm = row["method"]
                for field, expected_value in runner.d.question_identity(video, qi, arm).items():
                    assert row[field] == expected_value
                c = windows[chosen[arm]]
                assert row["eligible"] == (len(windows) > 1) and row["shard"] == video["shard"]
                assert all(row[k] == c[k] for k in ("window", "indices", "pred_choice", "correct", "query_kv_bytes"))
                assert math.isfinite(row["component_s"]) and row["component_s"] >= 0
                probes = runner.ARM_PROBES[arm]
                scanned = windows if probes and len(windows) > 1 else [c]
                seconds = c["answer_s"] + sum(w["construction_s"] + sum(w["probe_s"].get(p, 0) for p in probes)
                                               for w in scanned)
                assert math.isclose(row["component_s"], seconds)
            assert block["oracle_correct"] == max(c["correct"] for c in windows)
            blocks.append(block)
    primary = [b for b in blocks if b["rows"][0]["eligible"]]
    completion = json.loads(payload["exports/completion.json"])
    assert completion["config_hash"] == config["config_hash"]
    assert completion["status"] == "complete"
    assert len(blocks) == completion["main_questions"] == manifest["questions"]
    assert len(primary) == manifest["primary_questions"]
    assert sum(len(b["rows"]) for b in blocks) == completion["main_answers"]
    assert json.loads(payload["preflight.json"])["status"] == "pass"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        runner.export(blocks, work, complete=True)
        for file in (work / "exports").iterdir():
            local = file.read_bytes()
            if file.suffix == ".md":
                local = local.replace(b"\r\n", b"\n")
            assert local == payload[f"exports/{file.name}"], file.name
    recent_by_older = defaultdict(int)
    for block in primary:
        recent, *older = block["windows"]
        recent_by_older[(bool(recent["correct"]), any(c["correct"] for c in older))] += 1
    text = [b for b in primary if b["rows"][0]["task"] == runner.TEXT_TASK]
    review = list(csv.DictReader(io.StringIO(payload["exports/blind_review.csv"].decode())))
    return dict(status="pass", archive_sha256=archive_sha, notebook_sha256=notebook_sha,
                payload_files=len(payload), videos=len(videos),
                questions=len(blocks), selector_outcomes=completion["main_answers"],
                candidate_windows=sum(len(b["windows"]) for b in blocks),
                elapsed_post_model_s=completion["elapsed_post_model_s"], primary_questions=len(primary),
                text_rich_primary_questions=len(text), media_bytes=sum(a["bytes"] for a in assets.values()),
                primary_correct={a:sum(b["rows"][i]["correct"] for b in primary) for i,a in enumerate(runner.ARMS)},
                primary_oracle_correct=sum(b["oracle_correct"] for b in primary),
                older_only_correct=recent_by_older[(False, True)], recent_only_correct=recent_by_older[(True, False)],
                both_correct=recent_by_older[(True, True)], neither_correct=recent_by_older[(False, False)],
                blind_review_pairs=len(review), paired=json.loads(payload["exports/paired.json"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=ARCHIVE)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    result = audit(args.archive, args.notebook)
    runner.p.atomic_json(AUDIT, result)
    print(f"Verified {result['payload_files']} payloads, {result['questions']} questions and {result['selector_outcomes']} outcomes.")
