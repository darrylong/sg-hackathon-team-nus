# Team <sg_hackathon_team_nus>

Members: <Ho Jone Mun, Koen Goh>

This folder is your team's hand-in. Keep this layout:

```
<your team>/
  README.md                 this file - fill it in
  slides/                   ONE slide deck (PDF or PPTX, about 6-10 slides) covering both problems
  03_ticket_triage/         submission files (overwrite them) + src/ (your code and full-stack MVP)
  05_channel_analytics/     submission files (overwrite them) + src/ (your code and full-stack MVP)
```

## How to run

| Problem | Start the MVP (one command) | Regenerate the submission files |
|---|---|---|
| 03 Ticket Triage | `cd 03_ticket_triage/src && python start.py` then open `http://127.0.0.1:8501` (API docs at `http://127.0.0.1:8000/docs`) | `cd 03_ticket_triage/src && python src/make_submission.py` (writes `outputs/final/`; copy `submission.csv` and `submission_curveball.csv` up into `03_ticket_triage/`) |
| 05 Channel Analytics | `cd 05_channel_analytics/src && CHANNEL_DATA_DIR=/path/to/pack/05_channel_analytics/data ./start.sh` then open `http://localhost:8000` | `cd 05_channel_analytics/src && .venv/bin/python -m src.run_all` (writes `submission/`; copy both CSVs up into `05_channel_analytics/`) |

Setup:

- **03 Ticket Triage:** Python 3.11+. Run `pip install -r requirements-app.txt` once, from `03_ticket_triage/src/`.
  Everything the MVP and the submission script need is in `src/`: the trained model bundle (`models/bundle_v3/`),
  the pack's `artifacts/` (cost matrix, capacity) and the `eval/` tickets. The pack's training data is not needed.
  Run the API tests with `python -m pytest tests -v`. Details are in `03_ticket_triage/README.md`.
- **05 Channel Analytics:** <...>

## Approach

### 03 Ticket Triage

Full write-up: `03_ticket_triage/README.md`.

- **Split:** 30,000 labelled tickets split once into TRAIN 21,000 / VAL 4,500 / TEST 4,500 (stratified, seed 42).
  Every model choice was made on VAL with a pre-written rule: paired bootstrap, 95% CI must exclude 0. TEST was
  scored once at the end.
- **Models (one per label, all reading `channel + product + subject + body` as text):**
  - Category: word TF-IDF + logistic regression.
  - Priority: ordinal model (three cumulative logistic regressions) on word + character n-gram TF-IDF. The label is
    P1 when P(P1) ≥ 0.225, which raises P1 recall on VAL from 0.60 to 0.81.
  - Team: gradient-boosted tree stacker over the category and team probabilities plus product/channel. It learns
    product exceptions such as switch firmware going to Network-Support.
  - Sentiment: word + character TF-IDF, chi² feature selection, gradient-boosted trees.
  - All probabilities are temperature-calibrated.
- **How we decide to abstain:** a correctness model predicts `p_correct`, the chance the routed team is right. For
  each ticket we compare the expected misroute cost, (1 − p_correct) × the priority-weighted misroute cost, with the
  expected hand-off cost from the cost matrix. Tickets where a hand-off is cheaper go to a human, highest savings
  first, up to the 15% cap. Two guards always force a hand-off: a product not seen in training, and an empty or very
  short body. Every decision comes with a plain-language reason.
- **Noisy labels:** we did not relabel or clean the training labels. Calibration keeps the probabilities honest, and
  the abstention rule sends low-confidence tickets to a human. Sentiment macro-F1 levels off at about 0.886 across
  every model we tried, which suggests label noise sets the ceiling.
- **Sarcasm and mixed languages:** no translation, embeddings or LLM. Character n-grams pick up product names, error
  codes and word fragments in any language. On the curveball set, the Malay (CB-003) and German (CB-011) tickets were
  routed to the correct team. Sarcasm is not handled: CB-001 ("Wow, what an achievement… rebooted itself four times")
  was scored as Neutral sentiment.

### 05 Channel Analytics
<key insights, how the health score works, forecast method and interval>

## Results

### 03 Ticket Triage

TEST hold-out (n = 4,500, scored once after all choices were locked on VAL):

| Metric | Result |
|---|---|
| Category / priority / team / sentiment macro-F1 | 0.862 / 0.769 / 0.898 / 0.886 |
| P1 precision / recall / F1 | 0.651 / 0.838 / 0.733 |
| Escalation-risk F1 | 0.858 |
| Routing cost, no abstention | 0.1700 |
| Routing cost, abstain on the 15% least confident | 0.1350 |
| **Routing cost, expected-cost abstention (ours)** | **0.1331 (9.7% handed off)** |

The four macro-F1 scores on TEST are within 0.007 of VAL, so selecting on VAL did not overfit noticeably.
On the eval file, 691 of 6,000 tickets (11.5%) are handed off; 2 of the 15 curveball tickets are handed off.

### 05 Channel Analytics
<your own hold-out metrics>

## Limitations and next steps

### 03 Ticket Triage

- **Unseen category only partly caught:** the eval set adds data-centre facilities tickets (cooling, PDUs, UPS,
  water leaks). About 195 eval tickets look like this, and only about 29 are handed off; most go to Server-HW (for
  example curveball CB-004, "server room aircon dead"). Next step: a facilities guard that tells room power and
  cooling apart from server PSU and fan faults.
- **Tickets with no real content still route** if they are over 20 characters, for example "same thing as before,
  it's doing it again" (CB-015). Next step: a guard for content-free tickets.
- **Sarcasm and non-English text** rely on character n-grams only. Next step: a multilingual sentence-embedding
  model as an extra feature.
- **Trained on 70% of the data:** the final models use TRAIN only. Next step: refit on all 30,000 rows.
- **Template-built data:** performance on real, free-form tickets will be lower.

### 05 Channel Analytics
<what does not work yet, and what you would do with more time>

## Tools and models used

- **03 Ticket Triage:** Python, scikit-learn, pandas, rapidfuzz, FastAPI, Streamlit. No pretrained models, no LLM
  or gateway; nothing is sent to external services. Claude Code was used as an AI coding assistant.
- **05 Channel Analytics:** <libraries, pretrained models, LLM / gateway use, AI coding assistants>
