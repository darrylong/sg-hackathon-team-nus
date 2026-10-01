"""Round 2A step 0: read-only diagnostics. Writes outputs/diagnostics/round2a.txt."""
import json

import pandas as pd

from data import ROOT, SPLITS, SPLIT_DIR, TARGETS, load_splits, load_tickets

OUT = ROOT / "outputs" / "diagnostics" / "round2a.txt"


def dup_report(df, cols, split_of):
    dup = df[df.duplicated(cols, keep=False)]
    groups = dup.groupby(cols)["ticket_id"].apply(list)
    spanning = sum(len({split_of[i] for i in ids}) > 1 for ids in groups)
    return (f"{'+'.join(cols)}: rows involved={len(dup)}, groups={len(groups)}, "
            f"groups spanning >1 split={spanning}")


def main():
    df = load_tickets()
    parts = load_splits(df)
    lines = []

    lines.append("== Label sets in TRAIN ==")
    for t in TARGETS:
        lines.append(f"{t}: {sorted(parts['train'][t].unique())}")

    split_of = {tid: s for s in SPLITS for tid in pd.read_csv(SPLIT_DIR / f"{s}_ids.csv")["ticket_id"]}
    lines.append("\n== Exact duplicates (all 30k rows) ==")
    lines.append(dup_report(df, ["subject", "body"], split_of))
    lines.append(dup_report(df, ["body"], split_of))

    preds = pd.read_csv(ROOT / "outputs" / "baseline" / "val_predictions.csv")
    fn = preds[(preds.true_sentiment == "Frustrated") & (preds.pred_sentiment == "Neutral")]
    sample = fn.sample(15, random_state=42).merge(df[["ticket_id", "channel", "subject", "body"]], on="ticket_id")
    lines.append(f"\n== 15 VAL tickets true Frustrated -> pred Neutral (of {len(fn)}) ==")
    for r in sample.itertuples():
        lines.append(f"[{r.ticket_id} | {r.channel} | conf={r.sentiment_confidence:.2f}] {r.subject}\n    {r.body[:300]}")

    lines.append("\n== artifacts/triage_capacity.json ==")
    lines.append(json.dumps(json.loads((ROOT / "artifacts" / "triage_capacity.json").read_text()), indent=2))
    lines.append("\n== artifacts/routing_cost_matrix.csv ==")
    lines.append(pd.read_csv(ROOT / "artifacts" / "routing_cost_matrix.csv")
                 .pivot(index="true_priority", columns="outcome", values="cost").to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
