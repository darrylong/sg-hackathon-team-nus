# Team <your team name>

Members: <names>

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
| 03 Ticket Triage | `<command>` then open `<url>` | `<command>` |
| 05 Channel Analytics | `cd 05_channel_analytics/src && CHANNEL_DATA_DIR=/path/to/pack/05_channel_analytics/data ./start.sh` then open `http://localhost:8000` | `cd 05_channel_analytics/src && .venv/bin/python -m src.run_all` (writes `submission/`; copy both CSVs up into `05_channel_analytics/`) |

Setup (install steps, where the code expects the pack's data): <...>

## Approach

### 03 Ticket Triage
<models, features, how you decide to abstain, how you handled noisy labels, sarcasm, mixed languages>

### 05 Channel Analytics
<key insights, how the health score works, forecast method and interval>

## Results

<your own hold-out metrics for each problem>

## Limitations and next steps

<what does not work yet, and what you would do with more time>

## Tools and models used

<libraries, pretrained models, LLM / gateway use, AI coding assistants>
