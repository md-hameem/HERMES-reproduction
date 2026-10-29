import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEY = ("video_id", "qa_index")
PINNED = {
    "reproduction_head": "af79580aab9c2e58b48432cc91088d374a0e0b04",
    "hermes_revision": "8d699b16a6bedb9086c1b39ec4253c6a1d1ce789",
    "transformers_revision": "66bc4def9505fa7c7fe4aa7a248c34a026bb552b",
    "model_snapshot": "74dd0bf867a4cda7950c17663794267c60cf4b40",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path, annotation_lookup):
    frame = pd.read_csv(path).copy()
    assert len(frame) == 250, (path, len(frame))
    assert frame.video_id.nunique() == 50, (path, frame.video_id.nunique())
    assert frame.groupby("video_id").size().eq(5).all(), path
    assert frame.video_id.notna().all()
    assert frame.question.notna().all()
    assert frame.qa_acc.notna().all()
    frame["video_id"] = frame.video_id.astype(str)
    frame["qa_index"] = frame.groupby("video_id", sort=False).cumcount().astype(int)
    assert not frame.duplicated(list(KEY)).any(), path

    for _, row in frame.iterrows():
        item = annotation_lookup[row.video_id]
        qa = item["conversations"][int(row.qa_index)]
        assert row.question.strip() == qa["question"].strip(), (path, row.video_id, row.qa_index)
        assert ast.literal_eval(row.choices) == qa["choices"], (path, row.video_id, row.qa_index)
        expected_letter = "ABCDEFGH"[qa["choices"].index(qa["answer"])]
        assert row.correct_choice == expected_letter, (path, row.video_id, row.qa_index)
        value = float(row.qa_acc)
        assert value in (0.0, 100.0), (path, row.video_id, row.qa_index)
        parsed = str(row.pred_choice).strip().upper() == expected_letter
        assert parsed == (value == 100.0), (path, row.video_id, row.qa_index, row.pred_choice)

    frame["correct"] = pd.to_numeric(frame.qa_acc).eq(100).astype(int)
    return frame


def verify_manifest(base):
    manifest = pd.read_csv(base / "exports/final/master_manifest_sha256.csv")
    assert len(manifest) == 53, len(manifest)
    assert manifest.path.is_unique
    assert set(manifest.origin) == {"preserved", "generated"}
    for _, row in manifest.iterrows():
        path = base / row.path
        assert path.is_file(), row.path
        assert path.stat().st_size == row.bytes, row.path
        assert sha256(path) == row.sha256, row.path
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    expected = set(manifest.path) | {"exports/final/master_manifest_sha256.csv"}
    assert actual == expected, (actual - expected, expected - actual)

    preserved = pd.read_csv(base / "preserved/exports/manifest_sha256.csv")
    assert len(preserved) == 16
    missing_preserved = []
    for _, row in preserved.iterrows():
        candidates = [base / "preserved" / row.path, base / row.path]
        matches = [path for path in candidates if path.is_file() and path.stat().st_size == row.bytes and sha256(path) == row.sha256]
        if not matches:
            missing_preserved.append(row.path)
    print(f"PASS master manifest: {len(manifest)}/53 payload files verified")
    print(f"PASS embedded preserved manifest: {len(preserved) - len(missing_preserved)}/16 files found with matching hashes")
    for path in missing_preserved:
        print(f"WARNING omitted from final evidence bundle: {path}")


def load_annotation(base):
    generated = json.loads((base / "data/streamingbench_50_autodl.json").read_text())
    preserved = json.loads((base / "preserved/HERMES/data/streamingbench_50_autodl.json").read_text())
    assert len(generated) == len(preserved) == 50
    assert sum(len(item["conversations"]) for item in generated) == 250
    assert sum(len(item["conversations"]) for item in preserved) == 250
    for new_item, old_item in zip(generated, preserved):
        assert new_item.keys() == old_item.keys()
        for field in new_item:
            if field == "video_path":
                assert "/root/autodl-tmp/hermes_final_new/" in new_item[field]
            else:
                assert new_item[field] == old_item[field], (new_item["video_id"], field)
    lookup = {str(item["video_id"]): item for item in generated}
    assert len(lookup) == 50
    return lookup


def parse_efficiency(method, path, outer_minutes):
    text = Path(path).read_text(errors="replace")
    ttft = [float(value) for value in re.findall(r"TTFT:\s*([0-9.eE+-]+)\s*seconds", text)]
    memory = [float(value) for value in re.findall(r"GPU memory usage:\s*([0-9.eE+-]+)\s*GB", text)]
    caches = [tuple(map(int, match)) for match in re.findall(r"Answering Cache lengths:\s*min=(\d+),\s*max=(\d+)", text)]
    chunks = [tuple(map(int, match)) for match in re.findall(r"Encoding frames\s+(\d+)\s+to\s+(\d+)", text)]
    fifo_events = text.count("FIFO_COMPRESSION")
    hermes_events = text.count("Applying KV-Cache compression")
    assert len(ttft) == 250, (method, len(ttft))
    assert len(memory) == 829, (method, len(memory))
    assert len(chunks) == 829, (method, len(chunks))
    assert len(caches) > 0, method
    assert fifo_events + hermes_events > 0, method
    assert not (fifo_events and hermes_events), method
    return {
        "method": method,
        "ttft_n": len(ttft),
        "median_ttft_s": float(np.median(ttft)),
        "mean_ttft_s": float(np.mean(ttft)),
        "max_reported_allocated_gib": max(memory),
        "memory_log_count": len(memory),
        "max_answering_cache": max(item[1] for item in caches),
        "compression_events": fifo_events + hermes_events,
        "compression_marker": "FIFO" if fifo_events else "HERMES",
        "encoded_chunks": len(chunks),
        "encoded_frames": sum(end - start + 1 for start, end in chunks),
        "outer_elapsed_min_this_session": outer_minutes,
    }


def paired_fifo_rows(raw):
    rows = []
    pairs = [
        ("FIFO_4000", "HERMES_4000"),
        ("FIFO_6000", "HERMES_6000"),
        ("FIFO_4000", "HERMES_50000"),
        ("FIFO_6000", "HERMES_50000"),
    ]
    for first, second in pairs:
        merged = raw[first][list(KEY) + ["correct"]].rename(columns={"correct": "a"}).merge(
            raw[second][list(KEY) + ["correct"]].rename(columns={"correct": "b"}),
            on=list(KEY),
            validate="one_to_one",
        )
        first_only = int(((merged.a == 1) & (merged.b == 0)).sum())
        second_only = int(((merged.a == 0) & (merged.b == 1)).sum())
        by_video = merged.assign(delta=merged.a - merged.b).groupby("video_id").delta.agg(["sum", "count"])
        sums = by_video["sum"].to_numpy(float)
        counts = by_video["count"].to_numpy(float)
        rng = np.random.default_rng(2024)
        draws = rng.integers(0, len(sums), size=(10000, len(sums)))
        boot = 100 * sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        rows.append(
            {
                "A": first,
                "B": second,
                "n": len(merged),
                "n_videos": merged.video_id.nunique(),
                "accuracy_A_pct": 100 * merged.a.mean(),
                "accuracy_B_pct": 100 * merged.b.mean(),
                "difference_A_minus_B_pp": 100 * (merged.a - merged.b).mean(),
                "A_only_correct": first_only,
                "B_only_correct": second_only,
                "video_cluster_CI95_low_pp": float(np.quantile(boot, 0.025)),
                "video_cluster_CI95_high_pp": float(np.quantile(boot, 0.975)),
                "exact_discordance_p_exploratory": binomtest(first_only, first_only + second_only, 0.5).pvalue
                if first_only + second_only
                else 1.0,
            }
        )
    return pd.DataFrame(rows)


def validate_notebook(notebook):
    data = json.loads(Path(notebook).read_text(encoding="utf-8"))
    outputs = [output for cell in data["cells"] for output in cell.get("outputs", [])]
    assert not any(output.get("output_type") == "error" for output in outputs)
    assert not any("traceback" in output for output in outputs)
    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    executed_cells = [cell for cell in code_cells if "".join(cell.get("source", [])).strip()]
    trailing_empty = [cell for cell in code_cells if not "".join(cell.get("source", [])).strip()]
    assert trailing_empty and all(cell.get("execution_count") is None for cell in trailing_empty)
    assert all(cell.get("execution_count") is not None for cell in executed_cells)
    counts = [cell["execution_count"] for cell in executed_cells]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    assert "DynamicCache.from_legacy_cache(past_key_values)" in source
    rendered = "\n".join("".join(output.get("text", [])) for output in outputs if output.get("name") == "stdout")
    assert "FULL SUITE INTEGRITY: PASS" in rendered
    assert "NVIDIA GeForce RTX 4080, 32760 MiB" in rendered
    assert "Transformers: 4.45.0.dev0 | pinned source: PASS" in rendered
    assert "Official HERMES source fidelity: PASS" in rendered
    assert "Model:" in rendered and "Snapshot identity: PASS" in rendered
    assert "preserved identity: PASS" in rendered
    assert "Both repeat generations are complete and validated." in rendered
    assert "Both FIFO generations completed and validated." in rendered
    assert "ATTENTION_DIFFERENTIAL_TEST_COMPLETE" in rendered
    archive_match = re.search(r"Archive SHA-256:\s*([0-9a-f]{64})", rendered)
    assert archive_match
    return archive_match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default="HERMES_AutoDL_final_evidence")
    parser.add_argument("--notebook", default="hermes-final-autodl.ipynb")
    args = parser.parse_args()
    base = Path(args.evidence_dir)

    verify_manifest(base)
    lookup = load_annotation(base)
    print("PASS annotation: generated and preserved question identities match exactly")

    method_paths = {
        "HERMES_4000": base / "preserved/HERMES/results/streamingbench_50_kv4000/1_0.csv",
        "HERMES_6000": base / "preserved/HERMES/results/streamingbench_50_kv6000/1_0.csv",
        "HERMES_50000": base / "preserved/HERMES/results/streamingbench_50_kv50000/1_0.csv",
        "HERMES_4000_REPEAT1": base / "results/streamingbench_50_kv4000_autodl_repeat1/1_0.csv",
        "HERMES_6000_REPEAT1": base / "results/streamingbench_50_kv6000_autodl_repeat1/1_0.csv",
        "FIFO_4000": base / "results/streamingbench_50_kv4000_fifo/1_0.csv",
        "FIFO_6000": base / "results/streamingbench_50_kv6000_fifo/1_0.csv",
    }
    raw = {name: read_predictions(path, lookup) for name, path in method_paths.items()}
    print("PASS raw predictions: 7 runs × 250 questions; source, gold, and parsed answers verified")

    names = list(raw)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            first = raw[first_name].set_index(list(KEY))
            second = raw[second_name].set_index(list(KEY))
            assert first.index.equals(second.index)
            for field in ("question", "choices", "answer", "correct_choice", "task"):
                assert first[field].fillna("").astype(str).equals(second[field].fillna("").astype(str)), (
                    first_name,
                    second_name,
                    field,
                )
    print("PASS cross-method metadata: identical positional question keys and labels")

    accuracy = pd.read_csv(base / "exports/final/master_accuracy.csv")
    expected_accuracy = pd.DataFrame(
        [
            {"method": name, "n": len(frame), "correct": int(frame.correct.sum()), "accuracy_pct": 100 * frame.correct.mean()}
            for name, frame in raw.items()
        ]
    )
    pd.testing.assert_frame_equal(accuracy[expected_accuracy.columns], expected_accuracy, check_exact=False, rtol=1e-12, atol=1e-12)
    print("PASS master accuracy: independently recomputed from raw CSVs")

    stability = pd.read_csv(base / "exports/final/repeatability_summary.csv")
    stability_rows = []
    for budget in (4000, 6000):
        before = raw[f"HERMES_{budget}"]
        after = raw[f"HERMES_{budget}_REPEAT1"]
        changed_choice = before.pred_choice.astype(str).str.strip().str.upper() != after.pred_choice.astype(str).str.strip().str.upper()
        merged = before[list(KEY) + ["correct"]].rename(columns={"correct": "before"}).merge(
            after[list(KEY) + ["correct"]].rename(columns={"correct": "after"}), on=list(KEY), validate="one_to_one"
        )
        stability_rows.append(
            {
                "kv_budget": budget,
                "n": len(merged),
                "n_videos": 50,
                "original_correct": int(merged.before.sum()),
                "repeat_correct": int(merged.after.sum()),
                "original_accuracy_pct": 100 * merged.before.mean(),
                "repeat_accuracy_pct": 100 * merged.after.mean(),
                "net_accuracy_change_pp": 100 * (merged.after - merged.before).mean(),
                "changed_choices": int(changed_choice.sum()),
                "changed_choice_pct": 100 * changed_choice.mean(),
                "changed_correctness": int((merged.before != merged.after).sum()),
                "changed_correctness_pct": 100 * (merged.before != merged.after).mean(),
                "became_correct": int(((merged.before == 0) & (merged.after == 1)).sum()),
                "became_wrong": int(((merged.before == 1) & (merged.after == 0)).sum()),
            }
        )
    pd.testing.assert_frame_equal(stability, pd.DataFrame(stability_rows), check_exact=False, rtol=1e-12, atol=1e-12)
    print("PASS repeatability: original/repeat changes independently recomputed")

    fifo_pairs = pd.read_csv(base / "exports/final/fifo_paired.csv")
    pd.testing.assert_frame_equal(fifo_pairs, paired_fifo_rows(raw), check_exact=False, rtol=1e-12, atol=1e-12)
    print("PASS FIFO paired tests: discordance statistics and video-cluster CIs reproduced")

    outer = {}
    original_runs = pd.read_csv(base / "exports/run_summary.csv")
    repeat_runs = pd.read_csv(base / "exports/repeat1/repeat1_runs.csv")
    fifo_runs = pd.read_csv(base / "exports/fifo/fifo_runs.csv")
    for budget, elapsed in zip(original_runs.kv_budget, original_runs.elapsed_min_this_session):
        outer[f"HERMES_{budget}"] = elapsed
    for budget, elapsed in zip(repeat_runs.kv_budget, repeat_runs.elapsed_min_this_session):
        outer[f"HERMES_{budget}_REPEAT1"] = elapsed
    for budget, elapsed in zip(fifo_runs.kv_budget, fifo_runs.elapsed_min_this_session):
        outer[f"FIFO_{budget}"] = elapsed

    log_paths = {
        "FIFO_4000": base / "logs/streamingbench_50_kv4000_fifo.log",
        "FIFO_6000": base / "logs/streamingbench_50_kv6000_fifo.log",
        "HERMES_4000": base / "preserved/logs/streamingbench_50_kv4000.log",
        "HERMES_6000": base / "preserved/logs/streamingbench_50_kv6000.log",
        "HERMES_50000": base / "preserved/logs/streamingbench_50_kv50000.log",
        "HERMES_4000_REPEAT1": base / "logs/streamingbench_50_kv4000_autodl_repeat1.log",
        "HERMES_6000_REPEAT1": base / "logs/streamingbench_50_kv6000_autodl_repeat1.log",
    }
    efficiency = pd.read_csv(base / "exports/final/master_efficiency.csv")
    expected_efficiency = pd.DataFrame([parse_efficiency(name, path, outer[name]) for name, path in log_paths.items()])
    pd.testing.assert_frame_equal(efficiency, expected_efficiency, check_exact=False, rtol=1e-12, atol=1e-12)
    print("PASS efficiency: all seven runs independently recomputed from raw logs")

    attention_layers = pd.read_csv(base / "exports/repeat1/attention/attention_per_layer_metrics.csv")
    attention_summary = pd.read_csv(base / "exports/repeat1/attention/attention_summary.csv")
    assert len(attention_layers) == 72
    assert attention_layers.layer.nunique() == 24
    assert set(attention_layers.comparison) == {
        "current_vs_native",
        "causal_frozen_vs_native",
        "current_vs_causal_frozen",
    }
    assert np.isfinite(attention_layers.select_dtypes("number").to_numpy()).all()
    attention_provenance = json.loads((base / "exports/repeat1/attention/attention_test_provenance.json").read_text())
    assert attention_provenance["model_snapshot"] == PINNED["model_snapshot"]
    assert attention_provenance["layers"] == 24
    assert attention_provenance["past_tokens"] == 13
    assert attention_provenance["query_tokens"] > 1
    expected_attention = (
        attention_layers.groupby("comparison")
        .agg(max_abs_max=("max_abs", "max"), mean_abs_mean=("mean_abs", "mean"), cosine_min=("cosine", "min"), cosine_mean=("cosine", "mean"))
        .reset_index()
        .merge(
            attention_layers.groupby("comparison")[[column for column in attention_layers if column.startswith("past_top")]].mean().reset_index(),
            on="comparison",
        )
    )
    pd.testing.assert_frame_equal(attention_summary, expected_attention, check_exact=False, rtol=1e-12, atol=1e-12)
    print("PASS attention differential: 24-layer metrics are finite and correctly summarized")

    for script_name in ("scripts/validate_pseudo_query_attention.py", "controls/fifo_vqa.py"):
        script = base / script_name
        compile(script.read_text(), str(script), "exec")
    fifo_source = (base / "controls/fifo_vqa.py").read_text()
    for forbidden in ("prune_kv_cache_by_attention(", "pseudo_forward(", "v_summary", "k_summary", "refined_scores"):
        assert forbidden not in fifo_source, forbidden
    for budget in (4000, 6000):
        text = (base / f"logs/streamingbench_50_kv{budget}_fifo.log").read_text(errors="replace")
        assert "FIFO_COMPRESSION" in text
    print("PASS controls: scripts compile; FIFO path excludes HERMES selection/summary logic")

    summary = json.loads((base / "exports/final/observed_research_summary.json").read_text())
    assert summary["reproduction_head"] == PINNED["reproduction_head"]
    assert summary["hermes_revision"] == PINNED["hermes_revision"]
    assert summary["transformers_revision"] == PINNED["transformers_revision"]
    assert summary["model_snapshot"] == PINNED["model_snapshot"]
    assert summary["original_inference_rerun"] is False
    assert summary["status"]["failed"] == []
    for source, target in ((accuracy, "accuracy"), (stability, "repeatability"), (fifo_pairs, "fifo_paired")):
        for index, row in source.iterrows():
            expected = summary[target][index]
            for column in source.columns:
                if pd.api.types.is_numeric_dtype(source[column]):
                    assert abs(float(expected[column]) - float(row[column])) < 1e-10, (target, index, column)
                else:
                    assert expected[column] == row[column], (target, index, column)
    print("PASS final summary/provenance: JSON exactly reflects source tables and pinned revisions")

    for relative in ("exports/final/figures/master_accuracy.png", "exports/final/figures/master_memory.png"):
        payload = (base / relative).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) > 10000
    print("PASS figures: both PNG artifacts are valid and non-empty")

    archive_hash = validate_notebook(args.notebook)
    print("PASS executed notebook: ordered executed cells, no errors, pinned environment/source/model verified")

    print("\nALL RESULT-INTEGRITY CHECKS PASSED; ARCHIVE WARNINGS ARE REPORTED ABOVE")
    print(f"Evidence directory: {base.resolve()}")
    print(f"Final manifest SHA-256: {sha256(base / 'exports/final/master_manifest_sha256.csv')}")
    print(f"Notebook-reported archive SHA-256: {archive_hash}")


if __name__ == "__main__":
    main()
