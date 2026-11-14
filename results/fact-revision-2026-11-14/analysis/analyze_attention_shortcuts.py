"""Descriptive query-rank and first/last-value analysis of certified controls.

Reads saved predictions and logits only. No training or inference is performed.
"""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import analyze_fact_revision as A

HERE = Path(__file__).resolve().parent
ARMS = ("attention32", "attention128")


def stats(rows, metric, label):
    return A.cluster_estimate(rows, metric, 5000, A.analysis_seed(78124, "posthoc:"+label))


def read_arm(root, arm, certified_manifest):
    run = root/arm
    for name in ("predictions.csv", "evaluation_data.jsonl.gz", "answer_logits.npy"):
        A.checked_hash(run/name, certified_manifest[arm+"/"+name]["sha256"], "unchanged certified input " + arm+"/"+name)
    episodes, current = {}, None
    for line in gzip.decompress((run/"evaluation_data.jsonl.gz").read_bytes()).decode().splitlines():
        item = json.loads(line)
        if item["type"] == "episodes":
            current = item["spec"]["cell"], item["spec"]["family"]
            for index, ep in enumerate(item["episodes"]):
                episodes[current+(index,)] = ep
    raw_rows = A.csv_rows(run/"predictions.csv")
    logits = np.load(run/"answer_logits.npy", allow_pickle=False)
    A.equal(logits.shape, (len(raw_rows), 256), "shortcut CSV/logit row count")
    rows = []
    for index, source in enumerate(raw_rows):
        ep = episodes[source["cell"], source["family"], int(source["episode"])]
        qi = int(source["query_index"])
        query = ep["queries"][qi]
        first, last = ep["events"][0], ep["events"][-1]
        pred, target = int(source["prediction"]), int(source["target"])
        A.equal(pred, int(np.argmax(logits[index])), "shortcut prediction/logit correspondence")
        A.equal(target, ord(query["target"]), "shortcut target from online history")
        first_value, last_value = ord(first["value"]), ord(last["value"])
        row = dict(arm=arm, cell=source["cell"], family=source["family"], condition=source["condition"],
            episode_seed=ep["seed"], episode=int(source["episode"]), query_index=qi,
            key=query["key"], query_template=query["template_index"], first_template=first["template_index"],
            last_template=last["template_index"], target=target, prediction=pred,
            first_value=first_value, last_value=last_value, distinct_values=first_value != last_value,
            correct=int(pred == target), first_match=int(pred == first_value), last_match=int(pred == last_value),
            first_heuristic_correct=int(first_value == target), last_heuristic_correct=int(last_value == target),
            first_only=int(pred == first_value and pred != last_value), last_only=int(pred == last_value and pred != first_value),
            both_values=int(pred == first_value == last_value), neither_value=int(pred not in (first_value, last_value)),
            legal_probability_mass=float(source["legal_probability_mass"]),
            old_value=ord(query["old_value"]) if query["old_value"] else -1,
            obsolete=int(query["old_value"] is not None and pred == ord(query["old_value"])),
            logits_row=index)
        rows.append(row)
    paired = []
    grouped = defaultdict(list)
    for row in rows:
        if row["cell"] in ("acquire_k2", "counterfactual_k2"):
            grouped[row["cell"], row["family"], row["condition"], row["episode_seed"]].append(row)
    for (cell, family, condition, seed), group in grouped.items():
        group.sort(key=lambda row: row["query_index"])
        A.equal([row["query_index"] for row in group], [0, 1], "two query ranks per K2 episode")
        first, second = group
        paired.append(dict(arm=arm, cell=cell, family=family, condition=condition, episode_seed=seed,
            distinct_values=first["distinct_values"], same_query_template=first["query_template"] == second["query_template"],
            first_query_template=first["query_template"], second_query_template=second["query_template"],
            different_predictions=int(first["prediction"] != second["prediction"]),
            both_correct=first["correct"]*second["correct"],
            first_query_correct=first["correct"], second_query_correct=second["correct"],
            query0_prediction=first["prediction"], query1_prediction=second["prediction"],
            max_abs_logit_difference=float(np.abs(logits[first["logits_row"]].astype(float)-logits[second["logits_row"]].astype(float)).max())))
    return rows, paired


def summarize(rows, pairs):
    ranks, sources, templates, pair_summary = [], [], [], []
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], row["cell"], row["family"], row["condition"]].append(row)
    for key, group in groups.items():
        arm, cell, family, condition = key
        for rank in sorted({row["query_index"] for row in group}):
            selected = [r for r in group if r["query_index"] == rank]
            result = stats(selected, "correct", repr((key, rank, "rank")))
            ranks.append(dict(arm=arm, cell=cell, family=family, condition=condition, query_index=rank,
                              correct=sum(r["correct"] for r in selected), **result))
        for subset in ("all", "different_values"):
            selected = group if subset == "all" else [r for r in group if r["distinct_values"]]
            if not selected:
                continue
            for metric in ("correct", "first_match", "last_match", "first_only", "last_only", "both_values", "neither_value",
                           "first_heuristic_correct", "last_heuristic_correct", "obsolete"):
                sources.append(dict(arm=arm, cell=cell, family=family, condition=condition, subset=subset,
                    metric=metric, count=sum(r[metric] for r in selected), **stats(selected, metric, repr((key, subset, metric)))))
        if cell in ("acquire_k2", "counterfactual_k2"):
            for qt in (0, 1):
                for rank in (0, 1):
                    selected = [r for r in group if r["query_template"] == qt and r["query_index"] == rank and r["distinct_values"]]
                    if not selected:
                        continue
                    templates.append(dict(arm=arm, cell=cell, family=family, condition=condition,
                        query_template=qt, query_index=rank, questions=len(selected),
                        correct=sum(r["correct"] for r in selected), first_only=sum(r["first_only"] for r in selected),
                        last_only=sum(r["last_only"] for r in selected), neither=sum(r["neither_value"] for r in selected)))
    grouped_pairs = defaultdict(list)
    for row in pairs:
        grouped_pairs[row["arm"], row["cell"], row["family"], row["condition"]].append(row)
    for key, group in grouped_pairs.items():
        for subset in ("all", "different_values", "different_values_same_query_template"):
            selected = [r for r in group if (subset == "all" or r["distinct_values"])
                        and (subset != "different_values_same_query_template" or r["same_query_template"])]
            if not selected:
                continue
            for metric in ("different_predictions", "both_correct"):
                pair_summary.append(dict(arm=key[0], cell=key[1], family=key[2], condition=key[3], subset=subset,
                    metric=metric, count=sum(r[metric] for r in selected), **stats(selected, metric, repr((key, subset, metric)))))
    return ranks, sources, templates, pair_summary


def select(table, **where):
    found = [r for r in table if all(str(r.get(k)) == str(v) for k, v in where.items())]
    A.equal(len(found), 1, "unique descriptive row " + repr(where))
    return found[0]


def number(row):
    return f"{100*float(row['estimate']):.2f}%"


def plot(out, estimates, ranks, sources, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.4), constrained_layout=True)
    colors = {"attention32": "#20639b", "attention128": "#914e9c"}
    labels = {"attention32": "Attention 32", "attention128": "Attention 128"}
    for arm_index, arm in enumerate(ARMS):
        offset = (-.11, .11)[arm_index]
        for index, cell in enumerate(("acquire_k1", "acquire_k2", "revise_k1")):
            row = select(estimates, arm=arm, cell=cell, family="seen", condition="online", metric="correct")
            point, low, high = [100*float(row[k]) for k in ("estimate", "ci_low", "ci_high")]
            axes[0, 0].errorbar(index+offset, point, yerr=[[point-low], [high-point]], fmt="o", capsize=4,
                color=colors[arm], label=labels[arm] if index == 0 else None)
        for rank in (0, 1):
            row = select(ranks, arm=arm, cell="acquire_k2", family="seen", condition="online", query_index=rank)
            point, low, high = [100*float(row[k]) for k in ("estimate", "ci_low", "ci_high")]
            axes[0, 1].errorbar(rank+offset, point, yerr=[[point-low], [high-point]], fmt="o", capsize=4, color=colors[arm])
            axes[0, 1].annotate(f"{row['correct']}/{row['queries']}", (rank+offset, point), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=8, color=colors[arm])
        for index, metric in enumerate(("different_predictions", "both_correct")):
            row = select(paired, arm=arm, cell="acquire_k2", family="seen", condition="online",
                         subset="different_values_same_query_template", metric=metric)
            point, low, high = [100*float(row[k]) for k in ("estimate", "ci_low", "ci_high")]
            axes[1, 1].errorbar(index+offset, point, yerr=[[point-low], [high-point]], fmt="o", capsize=4, color=colors[arm])
            axes[1, 1].annotate(f"{row['count']}/{row['episodes']}", (index+offset, point), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=8, color=colors[arm])
    axes[0, 0].set_xticks(range(3), ["One fact\n64 queries", "Two facts\n128 queries", "One revision\n64 queries"])
    axes[0, 0].set_title("Short-task performance (seen templates)")
    axes[0, 0].set_ylabel("Full-vocabulary accuracy (%)")
    axes[0, 0].axhline(80, color="#888888", ls="--", lw=1)
    axes[0, 0].text(.97, .82, "80% target", transform=axes[0, 0].transAxes,
                    ha="right", fontsize=8, color="#777777")
    axes[0, 0].legend(loc="upper center", fontsize=8)
    axes[0, 1].set_xticks([0, 1], ["Query first fact", "Query second fact"])
    axes[0, 1].set_title("Two-fact accuracy by queried position")
    axes[0, 1].set_ylabel("Full-vocabulary accuracy (%)")
    source_colors = {"first_only": "#3b966b", "last_only": "#df9d3e", "neither_value": "#c1c4c8"}
    source_labels = {"first_only": "First value", "last_only": "Last value", "neither_value": "Neither value"}
    for index, arm in enumerate(ARMS):
        bottom = 0
        for metric in ("first_only", "last_only", "neither_value"):
            row = select(sources, arm=arm, cell="acquire_k2", family="seen", condition="online",
                         subset="different_values", metric=metric)
            value = 100*float(row["estimate"])
            axes[1, 0].bar(index, value, bottom=bottom, color=source_colors[metric], width=.55,
                           label=source_labels[metric] if index == 0 else None)
            if value >= 5:
                axes[1, 0].text(index, bottom+value/2, f"{row['count']}/{row['queries']}", ha="center", va="center", fontsize=9)
            bottom += value
    axes[1, 0].set_xticks([0, 1], [labels[a] for a in ARMS])
    axes[1, 0].set_title("Which presented value is predicted?\nOnly episodes with different fact values")
    axes[1, 0].set_ylabel("Questions (%)")
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=8)
    axes[1, 1].set_xticks([0, 1], ["Prediction changes", "Both answers correct"])
    axes[1, 1].set_title("Change only the queried key\nSame query wording; different fact values")
    axes[1, 1].set_ylabel("Eligible episodes (%)")
    for axis in axes.flat:
        axis.set_ylim(0, 108 if axis is axes[0, 1] or axis is axes[1, 1] else 100)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    axes[0, 1].set_xlim(-.5, 1.5)
    axes[1, 1].set_xlim(-.5, 1.5)
    fig.suptitle("Attention controls: descriptive position and value-choice diagnostics", fontsize=15)
    fig.supxlabel("One fixed parent; trained only on two distinct facts. One-fact and revision cells test transfer.\n95% intervals resample whole episodes. This follow-up does not rescue the failed main acquisition gate.", fontsize=9)
    fig.savefig(out/"attention_controls_diagnostics.png", dpi=180)
    fig.savefig(out/"attention_controls_diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures-out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root, audited, out = args.root.resolve(), args.audit.resolve(), args.out.resolve()
    A.require(out.is_relative_to(HERE) and out != HERE, "Output must be a new analysis subdirectory")
    figures = args.figures_out.resolve() if args.figures_out else out
    A.require(figures.is_relative_to(HERE) and figures != HERE, "Figures must stay under the analysis directory")
    verify = A.read_json(audited/"verification.json")
    A.require(verify.get("status") == "verified_supplement" and verify.get("supplemental_certified") is True,
              "Control artifacts must first pass the separate auditor")
    for name in ("estimates.csv", "contrasts.csv", "decisions.json", "source-manifest.json"):
        A.checked_hash(audited/name, verify["output_sha256"][name], "verified supplemental table " + name)
    manifest = A.read_json(audited/"source-manifest.json")["artifacts"]
    all_rows, all_pairs = [], []
    for arm in ARMS:
        rows, pairs = read_arm(root, arm, manifest)
        all_rows.extend(rows)
        all_pairs.extend(pairs)
    ranks, sources, templates, paired = summarize(all_rows, all_pairs)
    estimates, contrasts = A.csv_rows(audited/"estimates.csv"), A.csv_rows(audited/"contrasts.csv")
    out.mkdir(parents=True, exist_ok=args.overwrite)
    if figures != out:
        figures.mkdir(parents=True, exist_ok=args.overwrite)
        # Relocate only our named plot outputs when splitting an existing report.
        for name in ("attention_controls_diagnostics.png", "attention_controls_diagnostics.pdf"):
            previous = out/name
            if previous.exists():
                previous.replace(figures/name)
    A.write_csv(out/"query_details.csv", all_rows)
    A.write_csv(out/"episode_responses.csv", all_pairs)
    A.write_csv(out/"query_rank_summary.csv", ranks)
    A.write_csv(out/"source_alignment.csv", sources)
    A.write_csv(out/"query_template_source_choices.csv", templates)
    A.write_csv(out/"episode_response_summary.csv", paired)
    summary = {}
    for arm in ARMS:
        common = dict(arm=arm, family="seen", condition="online")
        summary[arm] = dict(
            acquisition=select(estimates, **common, cell="acquire_k2", metric="correct"),
            legal_mass=select(estimates, **common, cell="acquire_k2", metric="legal_probability_mass"),
            rank0=select(ranks, **common, cell="acquire_k2", query_index=0),
            rank1=select(ranks, **common, cell="acquire_k2", query_index=1),
            first_alignment=select(sources, **common, cell="acquire_k2", subset="different_values", metric="first_only"),
            last_alignment=select(sources, **common, cell="acquire_k2", subset="different_values", metric="last_only"),
            neither_alignment=select(sources, **common, cell="acquire_k2", subset="different_values", metric="neither_value"),
            fixed_first_heuristic=select(sources, **common, cell="acquire_k2", subset="all", metric="first_heuristic_correct"),
            fixed_last_heuristic=select(sources, **common, cell="acquire_k2", subset="all", metric="last_heuristic_correct"),
            query_changes=select(paired, **common, cell="acquire_k2", subset="different_values_same_query_template", metric="different_predictions"),
            both_correct=select(paired, **common, cell="acquire_k2", subset="different_values_same_query_template", metric="both_correct"),
            different_value_pairs=select(paired, **common, cell="acquire_k2", subset="different_values", metric="different_predictions"),
            revision_accuracy=select(estimates, **common, cell="revise_k1", metric="correct"),
            revision_old=select(sources, **common, cell="revise_k1", subset="all", metric="obsolete"),
            revision_other=select(sources, **common, cell="revise_k1", subset="all", metric="neither_value"),
            one_fact_accuracy=select(estimates, **common, cell="acquire_k1", metric="correct"))
    primary = select(contrasts, cell="acquire_k2", family="seen", condition="online", metric="correct")
    report = dict(analysis_role="Posthoc descriptive behavior analysis of certified saved control outputs", summary=summary,
        paired_attention128_minus32=primary,
        limits=["One fixed parent and one training data seed; no claim about training-procedure variance.",
            "Heuristic alignment describes outputs and does not establish the model's internal mechanism.",
            "Same-wording paired questions change only the requested key while keeping the two facts fixed.",
            "One-fact and revision cells differ from the K2/no-revision-only training distribution.",
            "Zero-event subsets have degenerate empirical-bootstrap intervals; zero observed events do not prove a zero population rate.",
            "Neither attention control replaces or rescues the main failed acquisition experiment."])
    A.write_json(out/"report.json", report)
    lines = ["# Descriptive attention-control behavior\n",
        "This is a posthoc analysis of independently certified saved outputs. No training or inference was rerun. Both controls used the same original checkpoint, 16,000 K2-only training examples, and the same 64 development episodes per short cell.\n",
        f"For two-fact acquisition, attention32 scored {number(summary['attention32']['acquisition'])} and attention128 scored {number(summary['attention128']['acquisition'])}. The paired attention128-minus32 difference was {100*float(primary['estimate']):+.2f} percentage points (95% episode-bootstrap interval {100*float(primary['ci_low']):+.2f} to {100*float(primary['ci_high']):+.2f}). Both remain below the descriptive 80% target.\n",
        "| Control | Query first fact | Query second fact | Legal-answer mass |\n|---|---:|---:|---:|"]
    for arm in ARMS:
        s = summary[arm]
        lines.append(f"| {arm} | {s['rank0']['correct']}/{s['rank0']['queries']} ({number(s['rank0'])}) | {s['rank1']['correct']}/{s['rank1']['queries']} ({number(s['rank1'])}) | {number(s['legal_mass'])} |")
    lines += ["\nBoth fixed-first and fixed-last value heuristics achieve " + number(summary["attention32"]["fixed_last_heuristic"]) + " over these balanced K2 queries; coincident values account for their slight advantage over 50%. Thus about 50% accuracy alone does not establish correct key–value binding.\n",
        "| Control | Predict first value | Predict last value | Neither |\n|---|---:|---:|---:|"]
    for arm in ARMS:
        s = summary[arm]
        lines.append(f"| {arm} | {s['first_alignment']['count']}/{s['first_alignment']['queries']} ({number(s['first_alignment'])}) | {s['last_alignment']['count']}/{s['last_alignment']['queries']} ({number(s['last_alignment'])}) | {s['neither_alignment']['count']}/{s['neither_alignment']['queries']} |")
    lines.append("\nThe value-choice table excludes episodes whose two fact values coincide. These are descriptive alignments, not identification of an internal selection rule.\n")
    for arm in ARMS:
        s = summary[arm]
        lines.append(f"For {arm}, among {s['query_changes']['episodes']} episodes with different fact values and identical query wording, changing only the requested key changed the top prediction in {s['query_changes']['count']} episodes; both queries were answered correctly in {s['both_correct']['count']}.\n")
        lines.append(f"Across all {s['different_value_pairs']['episodes']} different-value episodes, predictions changed in {s['different_value_pairs']['count']}; this wider set also permits query wording to change, so it does not isolate a requested-key change.\n")
    lines.append("The zero-event subsets have degenerate empirical-bootstrap intervals. Their observed counts do not establish a zero event probability beyond this development sample.\n")
    lines += ["| Control | Revision: current value | Revision: obsolete value | Neither value | One-fact accuracy |\n|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = summary[arm]
        lines.append(f"| {arm} | {number(s['revision_accuracy'])} | {s['revision_old']['count']}/{s['revision_old']['queries']} ({number(s['revision_old'])}) | {s['revision_other']['count']}/{s['revision_other']['queries']} | {number(s['one_fact_accuracy'])} |")
    lines.append("\nThe controls trained only on two distinct facts with no revisions. One-fact inputs also have a different length (63 rather than 95 bytes), while revisions reuse one key. Their results measure transfer beyond that training recipe. The observed behavior does not establish reliable arbitrary-key acquisition or revision handling, and does not rescue the main replay experiment.\n")
    lines.append("All estimates and subsets are saved in CSV, including heldout templates, reset conditions, counterfactual assignments, query-template strata, and episode responses. The intervals condition on this fixed parent and development sample.\n")
    (out/"report.md").write_text("\n".join(lines), encoding="utf8")
    plot(figures, estimates, ranks, sources, paired)
    files = [p for p in out.iterdir() if p.is_file() and p.name != "manifest.json"]
    A.write_json(figures/"plot-manifest.json", dict(script_path=str(Path(__file__).resolve()), script_sha256=A.sha(__file__),
        source_tables={name: dict(path=str(out/name), sha256=A.sha(out/name))
                       for name in ("query_rank_summary.csv", "source_alignment.csv", "episode_response_summary.csv", "report.json")},
        certified_estimates=dict(path=str(audited/"estimates.csv"), sha256=A.sha(audited/"estimates.csv")),
        outputs={name: A.sha(figures/name) for name in ("attention_controls_diagnostics.png", "attention_controls_diagnostics.pdf")}))
    A.write_json(out/"manifest.json", dict(script_sha256=A.sha(__file__), audit_verification_sha256=A.sha(audited/"verification.json"),
        audited_inputs={arm+"/"+name: manifest[arm+"/"+name] for arm in ARMS
                        for name in ("predictions.csv", "evaluation_data.jsonl.gz", "answer_logits.npy")},
        outputs={p.name: A.sha(p) for p in files}, figures_directory=str(figures),
        figure_manifest_sha256=A.sha(figures/"plot-manifest.json")))
    print(json.dumps(dict(status="completed_descriptive_analysis", out=str(out), summary=summary,
                          paired_attention128_minus32=primary)), flush=True)


if __name__ == "__main__":
    main()
