# agentic_plotter

Chat-based scientific data exploration app (Dash + Plotly) with LangGraph-backed analysis sessions.

## What it does
- Load a CSV/Parquet catalogue where each row is a dataset/event.
- Ask natural-language questions like: *"Plot all category 2 events as a scatter plot with event duration on x and mean event current on y."*
- The app converts your request into a validated `PlotSpec` (Pydantic), renders an interactive Plotly chart, and stores the interaction/recipe in a persistent session.

## Setup (Windows PowerShell)
1. Create/activate a venv (you already have `env0/` in the repo)
2. Install deps:
   - `pip install -r agentic_plotter/requirements.txt`
3. Set your OpenAI key:
   - Create `agentic_plotter/.env` with:
     - `OPENAI_API_KEY=...`

## Run
- `python -m agentic_plotter.app`
- Open the URL printed in the terminal.

## Internal smoke test (no OpenAI needed)
This helps confirm that the catalogue loads and plotting works using your real file:
- `C:\hackathon\Data\data_1.csv`

Run:
- `python -m agentic_plotter.tests.smoke_test`

If this passes but the web app fails, the bug is likely in callbacks / LangGraph / OpenAI call.

## Notes
- Session data is stored under `agentic_plotter/.sessions/` (JSON).
- The LLM output is *strictly validated* into `PlotSpec`. If invalid, the assistant will ask clarifying questions.
