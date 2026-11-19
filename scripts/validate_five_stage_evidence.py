"""Independently audit the preserved five-stage ZIP without executing its code."""

import argparse
import ast
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
KEY = ["video_id", "qa_index"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity(records):
    return [{"video_id": str(x["video_id"]), "conversations": x["conversations"]} for x in records]


def sample_id(record):
    return int(re.search(r"sample_(\d+)", record["video_path"])[1])


def check_equal(actual, expected):
    if expected is None:
        assert actual is None or pd.isna(actual), (actual, expected)
    elif isinstance(expected, (float, int, np.number)):
        assert np.isclose(float(actual), expected, rtol=1e-9, atol=1e-9), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def predictions(data, records):
    frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    frame["qa_index"] = frame.groupby("video_id", sort=False).cumcount()
    lookup = {(str(v["video_id"]), i): q for v in records for i, q in enumerate(v["conversations"])}
    assert len(frame) == len(lookup) and not frame.duplicated(KEY).any()
    assert set(map(tuple, frame[KEY].values)) == set(lookup)
    for row in frame.itertuples():
        qa = lookup[(row.video_id, row.qa_index)]
        assert row.question.strip() == str(qa["question"]).strip()
        assert ast.literal_eval(row.choices) == qa["choices"]
        assert row.answer == str(qa["answer"])
        gold = "ABCDEFGH"[qa["choices"].index(qa["answer"])]
        assert row.correct_choice == gold
        # Preserve the original parser and first-match gold convention used by inference.
        text = row.pred_answer.strip()
        parsed = text[text.index(")") - 1:text.index(")")] if ")" in text else text[:1]
        assert parsed == row.pred_choice, (row.video_id, row.qa_index, parsed, row.pred_choice)
        assert float(row.qa_acc) in (0, 100)
        assert (row.pred_choice.strip().upper() == gold) == (float(row.qa_acc) == 100)
        if "task" in frame and "task" in qa:
            assert row.task == qa["task"]
    frame["correct"] = pd.to_numeric(frame.qa_acc).eq(100).astype(int)
    return frame


def paired(a, b):
    joined = a[KEY + ["correct"]].merge(b[KEY + ["correct"]], on=KEY, validate="one_to_one")
    wins = int(((joined.correct_x == 1) & (joined.correct_y == 0)).sum())
    losses = int(((joined.correct_x == 0) & (joined.correct_y == 1)).sum())
    groups = joined.assign(delta=joined.correct_x - joined.correct_y).groupby("video_id").delta.agg(["sum", "count"])
    draws = np.random.default_rng(2024).integers(0, len(groups), size=(10000, len(groups)))
    bootstrap = 100 * groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(axis=1)
    return dict(n=len(joined), n_videos=len(groups), accuracy_A_pct=100 * joined.correct_x.mean(),
                accuracy_B_pct=100 * joined.correct_y.mean(), difference_A_minus_B_pp=100 * (joined.correct_x - joined.correct_y).mean(),
                A_only_correct=wins, B_only_correct=losses,
                video_cluster_CI95_low_pp=float(np.quantile(bootstrap, .025)),
                video_cluster_CI95_high_pp=float(np.quantile(bootstrap, .975)),
                exact_discordance_p_exploratory=float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.)


def event_totals(events):
    totals = {}
    for name in ("encode", "pseudo_forward", "prune", "answer", "compression_path"):
        values = [x["seconds"] for x in events if x["event"] == name]
        assert all(np.isfinite(x) and x >= 0 for x in values)
        totals.update({name + "_n": len(values), name + "_s": sum(values),
                       name + "_median_s": float(np.median(values)) if values else None,
                       name + "_p95_s": float(np.quantile(values, .95)) if values else None})
    totals["encoded_frames"] = sum(x.get("frames", 0) for x in events if x["event"] == "encode")
    totals["max_cache_tokens"] = max(x["cache_max"] for x in events if "cache_max" in x)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/archives/HERMES_AutoDL_five_stage_evidence.zip")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/audits/five_stage_local_audit.json")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        payload = {name: archive.read(name) for name in archive.namelist()}
        assert len(payload) == len(archive.infolist())
    def table(name):
        return pd.read_csv(io.BytesIO(payload[name]))
    def record(name):
        return json.loads(payload[name])

    manifest_name = "exports/five_stage_manifest_sha256.csv"
    manifest = table(manifest_name)
    assert manifest.path.is_unique and set(payload) == set(manifest.path) | {manifest_name}
    for row in manifest.itertuples():
        assert len(payload[row.path]) == row.bytes and digest(payload[row.path]) == row.sha256, row.path
    published = ROOT / "results/five_stage"
    for path in published.rglob("*"):
        if path.is_file():
            archive_path = "exports/" + path.relative_to(published).as_posix()
            assert path.read_bytes() == payload[archive_path], path
    prior_manifest = table("preserved_final/exports/final/master_manifest_sha256.csv")
    for row in prior_manifest.itertuples():
        data = payload["preserved_final/" + row.path]
        assert digest(data) == row.sha256
        assert data == (ROOT / "evidence/autodl_2026-10-28" / row.path).read_bytes(), row.path
    old_manifest = table("preserved_final/preserved/exports/manifest_sha256.csv")
    omitted = [row for row in old_manifest.itertuples() if "preserved_final/preserved/" + row.path not in payload and "preserved_final/" + row.path not in payload]
    assert len(omitted) == 3
    with ZipFile(ROOT / "evidence/archives/historical_evidence_supplement.zip") as supplement:
        for row in omitted:
            assert digest(supplement.read(row.path)) == row.sha256
    print(f"PASS archive manifest {len(manifest)} payloads; all {len(prior_manifest)} prior payloads match October evidence", flush=True)

    annotations = json.loads((ROOT / "data/streamingbench/streamingbench_realtime.json").read_text())
    assert len(annotations) == 498 and sum(len(x["conversations"]) for x in annotations) == 2495
    assert len({str(v["video_id"]) for v in annotations}) == 498
    shards = {}
    shard_counts = []
    for start in range(1, 501, 50):
        end = start + 49
        expected = sorted((x for x in annotations if start <= sample_id(x) <= end), key=sample_id)
        actual = record(f"data/streamingbench_{start}_{end}.json")
        assert identity(actual) == identity(expected), start
        assets = record(f"data/streamingbench_{start}_{end}_assets.json")
        assert assets["status"] == "complete" and len(assets["videos"]) == len(actual)
        shards[f"{start}_{end}"] = actual
        shard_counts.append(dict(shard=f"{start}-{end}", videos=len(actual), questions=sum(len(x["conversations"]) for x in actual)))

    executed_bytes = args.notebook.read_bytes()
    notebook = json.loads(executed_bytes)
    assert not any(o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", []))
    rendered = "\n".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    archive_hash = digest(args.archive.read_bytes())
    for marker in ("FULL BENCHMARK COMPLETE", "FIVE-STAGE SUITE INTEGRITY: PASS", archive_hash):
        assert marker in rendered, marker
    cell = next("".join(c["source"]) for c in notebook["cells"] if "RUNNER.write_text(r'''" in "".join(c["source"]))
    writes = [x.value for x in ast.parse(cell).body if isinstance(x, ast.Expr) and isinstance(x.value, ast.Call) and isinstance(x.value.func, ast.Attribute) and x.value.func.attr == "write_text"]
    original_runner = next(x.args[0].value.encode() for x in writes if x.func.value.id == "RUNNER")
    numeric_fix = original_runner.replace(b"frame = pd.read_csv(path)", b"frame = pd.read_csv(path, dtype=str, keep_default_na=False)")
    final_runner = numeric_fix.replace(b"assert choices and answer in choices and choices.count(answer) == 1", b"assert choices and answer in choices")
    assert final_runner == payload["scripts/experiment_runner.py"]
    assert numeric_fix == payload["scripts/experiment_runner.py.before_validation_fix"]
    runners = {digest(x) for x in (original_runner, numeric_fix, final_runner)}
    for name, data in payload.items():
        if name.startswith("scripts/") and name.endswith(".py"):
            compile(data, name, "exec")
    print("PASS notebook completion/hash and three runner revisions differ only by recorded validator fixes", flush=True)

    prior = {}
    prior_paths = {
        "HERMES_4000": "preserved/HERMES/results/streamingbench_50_kv4000/1_0.csv",
        "HERMES_6000": "preserved/HERMES/results/streamingbench_50_kv6000/1_0.csv",
        "HERMES_50000": "preserved/HERMES/results/streamingbench_50_kv50000/1_0.csv",
        "HERMES_4000_REPEAT1": "results/streamingbench_50_kv4000_autodl_repeat1/1_0.csv",
        "HERMES_6000_REPEAT1": "results/streamingbench_50_kv6000_autodl_repeat1/1_0.csv",
        "FIFO_4000": "results/streamingbench_50_kv4000_fifo/1_0.csv",
        "FIFO_6000": "results/streamingbench_50_kv6000_fifo/1_0.csv"}
    for method, name in prior_paths.items():
        prior[method] = predictions(payload["preserved_final/" + name], shards["201_250"])
    runs, checkpoint_count = {}, 0
    for name in sorted(x for x in payload if x.startswith("results/") and x.endswith("/run_complete.json")):
        directory = name.rsplit("/", 1)[0]
        records = shards["251_300"] if "/full_benchmark/" not in directory else shards[directory.split("/")[2]]
        config = record(directory + "/config.json")
        config_hash = config["config_hash"]
        assert digest(stable_json({k: v for k, v in config.items() if k != "config_hash"}).encode()) == config_hash
        assert config["runner_sha256"] in runners
        assert config["annotation_sha256"] == digest(stable_json(identity(records)).encode())
        for source, sha in config["source_sha256"].items():
            assert digest((ROOT / source).read_bytes().replace(b"\r\n", b"\n")) == sha, source
        full = predictions(payload[directory + "/1_0.csv"], records)
        video_frames = []
        for video in records:
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video["video_id"]))
            frame = predictions(payload[directory + "/videos/" + stem + ".csv"], [video])
            metric = record(directory + "/video_metrics/" + stem + ".json")
            assert metric["status"] == "complete" and metric["config_hash"] == config_hash
            assert metric["questions"] == len(frame) and metric["correct"] == int(frame.correct.sum())
            for key, value in event_totals(metric["events"]).items():
                check_equal(metric[key], value)
            video_frames.append(frame)
            checkpoint_count += 1
        summary = table(directory + "/video_summary.csv")
        assert len(summary) == len(records)
        for row in summary.to_dict("records"):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["video_id"]))
            metric = record(directory + "/video_metrics/" + stem + ".json")
            for key, value in row.items():
                check_equal(str(value) if key == "video_id" else value, metric[key])
        combined = pd.concat(video_frames, ignore_index=True)
        pd.testing.assert_frame_equal(full.sort_values(KEY).reset_index(drop=True), combined[full.columns].sort_values(KEY).reset_index(drop=True), check_dtype=False)
        status = record(name)
        for key, value in dict(status="complete", config_hash=config_hash, videos=len(records), questions=len(full), correct=int(full.correct.sum()), accuracy_pct=100 * full.correct.mean()).items():
            check_equal(status[key], value)
        log_name = directory.split("/")[-2] + "_" + directory.split("/")[-1] + ".log"
        log = payload["logs/" + log_name].decode("utf8", errors="replace")
        completion = json.loads(re.findall(r"RUN_COMPLETE=(\{[^\n]+\})", log)[-1])
        assert completion == status
        runs[directory] = full
    print(f"PASS {len(runs)} new runs, {checkpoint_count} video checkpoints, event totals, config/source hashes and completion logs", flush=True)

    general = {k.rsplit("/", 1)[1].upper().replace("_KV", "_"): v for k, v in runs.items() if "/generalization_" in k}
    ablations = {k.rsplit("/", 1)[1].upper().replace("_KV", "_"): v for k, v in runs.items() if "/ablations_" in k}
    full = {}
    for method in ("HERMES", "FIFO"):
        pieces = [prior[method + "_4000"], general[method + "_4000"]]
        pieces.extend(v for k, v in runs.items() if "/full_benchmark/" in k and k.endswith("/" + method.lower() + "_kv4000"))
        combined = pd.concat(pieces, ignore_index=True)
        exported = predictions(payload[f"exports/full_{method.lower()}_4000_predictions.csv"], annotations)
        pd.testing.assert_frame_equal(exported.sort_values(KEY).reset_index(drop=True), combined[exported.columns].sort_values(KEY).reset_index(drop=True), check_dtype=False)
        full[method + "_4000"] = exported
    scopes = dict(prior_201_250=prior, generalization_251_300=general, ablation_251_300=ablations, full_498=full)
    accuracy = table("exports/master_accuracy_all_stages.csv")
    assert len(accuracy) == sum(map(len, scopes.values()))
    for row in accuracy.itertuples():
        frame = scopes[row.scope][row.method]
        assert row.videos == frame.video_id.nunique() and row.questions == len(frame) and row.correct == frame.correct.sum()
        check_equal(row.accuracy_pct, 100 * frame.correct.mean())
    for file, data in (("generalization_paired", general), ("ablation_paired", {**general, **ablations}), ("full_benchmark_paired", full)):
        for row in table("exports/" + file + ".csv").to_dict("records"):
            for key, value in paired(data[row["A"]], data[row["B"]]).items():
                check_equal(row[key], value)
    for row in table("exports/accuracy_by_task.csv").itertuples():
        data = {"generalization": general, "ablations": ablations, "full": full}[row.scope][row.method]
        subset = data[data.task.eq(row.task)]
        assert row.questions == len(subset) and row.correct == subset.correct.sum()
        check_equal(row.accuracy_pct, 100 * subset.correct.mean())
    print("PASS full aggregation, 18 accuracy rows, task tables and all paired tests / 10000 video bootstrap replicates", flush=True)

    efficiency_rows, event_rows = [], []
    for name in sorted(x for x in payload if x.startswith("efficiency_interleaved/repeat") and x.endswith("/metrics.json")):
        metric = record(name)
        video = next(v for v in shards["251_300"] if str(v["video_id"]) == metric["video_id"])
        frame = predictions(payload[name.replace("metrics.json", "predictions.csv")], [video])
        assert metric["questions"] == len(frame) and metric["correct"] == int(frame.correct.sum())
        assert metric["status"] == "complete"
        for key, value in event_totals(metric["events"]).items():
            check_equal(metric[key], value)
        efficiency_rows.append({k: v for k, v in metric.items() if k != "events"})
        for event in metric["events"]:
            event_rows.append(dict(repetition=metric["repetition"], video_id=metric["video_id"], method=metric["method"], **event))
    eff = pd.DataFrame(efficiency_rows)
    assert len(eff) == 60 and not eff.duplicated(["repetition", "video_id", "method"]).any()
    assert eff.groupby("method").size().to_dict() == {"fifo": 30, "hermes": 30}
    exported_eff = table("exports/efficiency_interleaved_video_metrics.csv")
    exported_eff["video_id"] = exported_eff.video_id.astype(str)
    sort = ["repetition", "video_id", "method"]
    pd.testing.assert_frame_equal(exported_eff.sort_values(sort).reset_index(drop=True), eff[exported_eff.columns].sort_values(sort).reset_index(drop=True), check_dtype=False, rtol=1e-9, atol=1e-9)
    exported_events = table("exports/efficiency_interleaved_event_metrics.csv")
    exported_events["video_id"] = exported_events.video_id.astype(str)
    events = pd.DataFrame(event_rows)
    order = sort + ["event", "seconds"]
    pd.testing.assert_frame_equal(exported_events.sort_values(order).reset_index(drop=True), events[exported_events.columns].sort_values(order).reset_index(drop=True), check_dtype=False, rtol=1e-9, atol=1e-9)
    for row in table("exports/efficiency_interleaved_summary.csv").to_dict("records"):
        data = eff[eff.method.eq(row["method"])]
        expected = dict(video_runs=len(data), questions=data.questions.sum(), correct=data.correct.sum(),
                        median_video_wall_s=data.wall_s.median(), p95_video_wall_s=data.wall_s.quantile(.95), total_wall_s=data.wall_s.sum(),
                        median_answer_s=data.answer_median_s.median(), total_encode_s=data.encode_s.sum(), encoded_frames=data.encoded_frames.sum(),
                        total_compression_path_s=data.compression_path_s.sum(), total_pseudo_forward_s=data.pseudo_forward_s.sum(), total_prune_s=data.prune_s.sum(),
                        max_cache_tokens=data.max_cache_tokens.max(), median_peak_allocated_gib=data.peak_allocated_gib.median(), max_peak_allocated_gib=data.peak_allocated_gib.max(),
                        median_peak_reserved_gib=data.peak_reserved_gib.median(), max_peak_reserved_gib=data.peak_reserved_gib.max(),
                        accuracy_pct=100 * data.correct.sum() / data.questions.sum(), frames_per_encode_s=data.encoded_frames.sum() / data.encode_s.sum())
        for key, value in expected.items():
            check_equal(row[key], value)
    for row in table("exports/efficiency_answer_latency_summary.csv").to_dict("records"):
        values = events[events.method.eq(row["method"]) & events.event.eq("answer")].seconds
        for key, value in dict(answer_events=len(values), median_answer_s=values.median(), mean_answer_s=values.mean(), p95_answer_s=values.quantile(.95), p99_answer_s=values.quantile(.99)).items():
            check_equal(row[key], value)
    timing = table("exports/efficiency_interleaved_paired_video.csv")
    assert len(timing) == 30 and not timing.duplicated(["repetition", "video_id"]).any()
    for row in timing.itertuples():
        data = eff[eff.repetition.eq(row.repetition) & eff.video_id.eq(str(row.video_id))].set_index("method")
        check_equal(row.fifo, data.loc["fifo", "wall_s"])
        check_equal(row.hermes, data.loc["hermes", "wall_s"])
        check_equal(row.hermes_minus_fifo_s, row.hermes - row.fifo)
        check_equal(row.fifo_over_hermes_ratio, row.fifo / row.hermes)
    print("PASS 60 interleaved video runs, all raw timing events, memory and latency summaries", flush=True)
    execution_order = table("efficiency_interleaved/execution_order.csv")
    assert len(execution_order) == 60 and not execution_order.duplicated(sort).any()
    execution_order["video_id"] = execution_order.video_id.astype(str)
    assert set(map(tuple, execution_order[sort].values)) == set(map(tuple, eff[sort].values))
    for _, group in execution_order.groupby(["repetition", "video_index"], sort=False):
        rep, index = int(group.iloc[0].repetition), int(group.iloc[0].video_index)
        assert group.method.tolist() == (["hermes", "fifo"] if (rep + index) % 2 == 0 else ["fifo", "hermes"])
    assert record("efficiency_interleaved/complete.json")["video_runs"] == 60
    shard_status = table("exports/full_benchmark_shard_status.csv")
    assert len(shard_status) == 20 and not shard_status.duplicated(["shard", "method"]).any()
    assert set(table("exports/generalization_run_status.csv").status) == {"completed"}

    attention = table("exports/real_video_attention/attention_per_layer_metrics.csv")
    assert len(attention) == 216 and np.isfinite(attention.select_dtypes("number")).all().all()
    expected = attention.groupby("comparison").agg(layers=("layer", "size"), mean_cosine=("cosine", "mean"), min_cosine=("cosine", "min"), mean_absolute_difference=("mean_absolute_difference", "mean"), max_absolute_difference=("max_absolute_difference", "max")).reset_index()
    pd.testing.assert_frame_equal(table("exports/real_video_attention/attention_summary.csv"), expected, rtol=1e-9, atol=1e-9)
    selection = table("exports/real_video_attention/selection_impact_per_layer.csv")
    assert len(selection) == 72
    for row in selection.itertuples():
        a, b = set(json.loads(row.selected_indices_A)), set(json.loads(row.selected_indices_B))
        assert len(a) == row.selected_A and len(b) == row.selected_B
        assert 0 < len(a) <= row.budget < row.visual_tokens
        assert row.intersection == len(a & b) and row.symmetric_difference == len(a ^ b)
        check_equal(row.jaccard, len(a & b) / len(a | b))
    expected = selection.groupby("comparison").agg(layers=("layer", "size"), mean_jaccard=("jaccard", "mean"), min_jaccard=("jaccard", "min"), mean_symmetric_difference=("symmetric_difference", "mean")).reset_index()
    pd.testing.assert_frame_equal(table("exports/real_video_attention/selection_impact_summary.csv"), expected, rtol=1e-9, atol=1e-9)
    provenance = record("exports/real_video_attention/attention_test_provenance.json")
    assert provenance["visual_tokens"] == 3136 and provenance["past_tokens"] == 3149
    assert provenance["layers"] == 24 and provenance["topk_budget_per_layer"] == 256
    for name, data in payload.items():
        if name.startswith("exports/figures/") and name.endswith(".png"):
            assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 10000
    print("PASS real-video attention tables and retained-token sets", flush=True)

    duplicates = [{"sample_id": sample_id(v), "video_id": str(v["video_id"]), "qa_index": i, "first_gold": "ABCDEFGH"[q["choices"].index(q["answer"])]}
                  for v in annotations for i, q in enumerate(v["conversations"]) if q["choices"].count(q["answer"]) > 1]
    warnings = [
        "Three historical payloads are absent from the embedded October bundle; their original hashes verify in evidence/archives/historical_evidence_supplement.zip.",
        "The executed notebook sources omit live patches; completion outputs and archived runner document recovery, so Run All reproducibility is not established for that snapshot.",
        "Full-benchmark results reuse October 201-250 outputs and November generalization outputs across sessions/hardware.",
        "One duplicate gold option in sample 463 uses first-match option A, matching the inference code.",
        "Native full-prefix LLaVA and a corrected-attention QA treatment remain unmeasured.",
        "Timing event scopes are nested; pseudo-query and pruning durations must not be added to compression-path time.",
    ]
    report = dict(status="PASS", archive_sha256=archive_hash, executed_notebook_sha256=digest(executed_bytes),
                  manifest_payloads=len(manifest), preserved_payloads=len(prior_manifest), new_complete_runs=len(runs),
                  validated_video_checkpoints=checkpoint_count, efficiency_video_runs=len(eff), shard_counts=shard_counts,
                  historical_supplement_sha256=digest((ROOT / "evidence/archives/historical_evidence_supplement.zip").read_bytes()),
                  full_benchmark=paired(full["HERMES_4000"], full["FIFO_4000"]),
                  duplicate_gold_options=duplicates, runner_revisions=sorted(runners), warnings=warnings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
    for warning in warnings:
        print("LIMIT " + warning)
    print("AUDIT PASS; " + str(args.output))


if __name__ == "__main__":
    main()
