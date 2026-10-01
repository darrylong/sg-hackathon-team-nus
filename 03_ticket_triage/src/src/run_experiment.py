"""Run experiments end to end: train (OOF + val) -> calibration -> correctness -> evaluate --ref baseline_lr.

python src/run_experiment.py <name> [<name> ...] [--retrain]
Names: keys of experiments.EXPERIMENTS, plus
  E3_tune_C        assemble the per-target C chosen by the selection rules from the E3_C_* runs
  E8_team_stacker  HGB stacker for team on the best run so far (by VAL policy-f cost)
  combined_v1      per-target best runs (COMBINED_SOURCES in experiments.py)
Prints one summary line per finished run; full output goes to the run folder.
"""
import argparse
import json
import subprocess
import sys
import traceback

import pandas as pd

import experiments as ex
from calibration import calibrate_run
from correctness import run_correctness
from data import ROOT, TARGETS, load_splits, load_tickets
from runs import RUNS_DIR, assemble_run, make_run

REF = "baseline_lr"
SRC = ROOT / "src"


def metrics(run):
    return json.loads((RUNS_DIR / run / "metrics.json").read_text())


def improves(run, target):
    """(is_significant_improvement, point_diff) vs baseline under the selection rules."""
    b = metrics(run)["bootstrap_vs_ref"]
    if target in ("assigned_team", "priority"):
        d = b["policy_f_cost"]
        return d["hi"] < 0, -d["diff"]
    d = b[f"{target}_macro_f1"]
    return d["lo"] > 0, d["diff"]


def evaluated_runs():
    return [p.name for p in RUNS_DIR.iterdir() if (p / "metrics.json").exists()
            and "bootstrap_vs_ref" in json.loads((p / "metrics.json").read_text())]


def best_run_by_cost_f(exclude=()):
    runs = [r for r in evaluated_runs() if r not in exclude] + [REF]
    return min(runs, key=lambda r: metrics(r)["routing"]["f_expcost_pcorrect"]["mean_cost"])


# ---------------------------------------------------------------- extra reports
def severity_report(run):
    train = load_splits(load_tickets())["train"]
    sev = train["body"].map(ex.parse_severity)
    ct = pd.crosstab(sev, train["priority"], margins=True)
    pct = pd.crosstab(sev, train["priority"], normalize="index").round(3)
    txt = f"Parsed severity vs TRUE priority (TRAIN)\n{ct}\n\nrow-normalised:\n{pct}\n\nby channel:\n" \
          f"{pd.crosstab(sev, train['channel'])}"
    (RUNS_DIR / run / "severity_crosstab.txt").write_text(txt)


def template_report(run):
    train = load_splits(load_tickets())["train"]
    tf = ex.TemplateFactors().fit(train)
    top = sorted(tf.counts_.items(), key=lambda kv: -kv[1])[:30]
    sents = train["body"].map(lambda b: set(ex.ticket_sentences(b)))
    rows = []
    for s, n in top:
        pr = train.loc[sents.map(lambda ss: s in ss), "priority"].value_counts(normalize=True)
        rows.append({"sentence": s[:90], "n_tickets": n, **{p: round(pr.get(p, 0), 3) for p in ["P1", "P2", "P3", "P4"]}})
    txt = f"kept sentences (>=20 TRAIN tickets): {len(tf.vocab_)}\n\n" + pd.DataFrame(rows).to_string(index=False)
    (RUNS_DIR / run / "template_sentences.txt").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------- build steps
def build(name):
    if name in ex.EXPERIMENTS:
        e = ex.EXPERIMENTS[name]
        make_run(name, e["make_model"], {"change": e["change"]}, base_run=REF)
        if name == "E2_severity_token":
            severity_report(name)
        if name == "E10_template_factors":
            template_report(name)
    elif name == "E3_tune_C":
        sources, chosen = {}, {}
        for t in TARGETS:
            cands = [(c, *improves(f"E3_C_{t}_{c}", t)) for c in ex.TUNE_C]
            sig = [(c, gain) for c, ok, gain in cands if ok]
            c_best = max(sig, key=lambda x: x[1])[0] if sig else 1
            chosen[t] = c_best
            sources[t] = REF if c_best == 1 else f"E3_C_{t}_{c_best}"
        assemble_run(name, sources, {"change": f"per-target C {chosen}", "chosen_C": chosen})
    elif name == "E8_team_stacker":
        src = best_run_by_cost_f(exclude={"E8_team_stacker", "combined_v1"})
        make_run(name, lambda t: None, {"change": f"HGB stacker for team on {src}", "source_run": src},
                 base_run=src, custom={"assigned_team": ex.team_stacker_producer(src)})
    elif name == "combined_v1":
        assemble_run(name, ex.COMBINED_SOURCES,
                     {"change": "per-target best: " + ", ".join(f"{t}<-{r}" for t, r in ex.COMBINED_SOURCES.items())})
    else:
        raise KeyError(name)


def change_of(name):
    cfg = RUNS_DIR / name / "config.json"
    return json.loads(cfg.read_text()).get("change", "") if cfg.exists() else ""


def summary_line(name):
    m = metrics(name)
    f = m["routing"]["f_expcost_pcorrect"]
    ci = m["bootstrap_vs_ref"]["policy_f_cost"]
    t = m["targets"]
    return (f"{name:28s} cat={t['category']['macro_f1']:.4f} prio={t['priority']['macro_f1']:.4f} "
            f"team={t['assigned_team']['macro_f1']:.4f} sent={t['sentiment']['macro_f1']:.4f} "
            f"P1rec={m['P1']['recall']:.3f} cost_b={m['routing']['b_naive_low_conf']['mean_cost']:.4f} "
            f"cost_f={f['mean_cost']:.4f} (abst {f['abstain_rate']:.3f}; vs base {ci['diff']:+.4f} "
            f"[{ci['lo']:+.4f},{ci['hi']:+.4f}])")


def run_one(name, retrain=False):
    if retrain or not (RUNS_DIR / name / "val.csv").exists():
        build(name)
    calibrate_run(name)
    run_correctness(name)
    res = subprocess.run([sys.executable, str(SRC / "evaluate.py"), "--run", name, "--ref", REF],
                         capture_output=True, text=True, cwd=ROOT)
    (RUNS_DIR / name / "evaluate_stdout.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"evaluate failed:\n{res.stderr[-2000:]}")
    print(summary_line(name), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    for n in args.names:
        try:
            run_one(n, args.retrain)
        except Exception:
            print(f"{n:28s} FAILED\n{traceback.format_exc()}", flush=True)
