# Medallion Pipeline

Architecture, the run-scoped artifact conventions, and the phase handoff table
are documented in [docs/architecture.md](docs/architecture.md).

An intent-driven pipeline that profiles CSV uploads, proposes layer mappings,
waits for human approval, materializes Bronze/Silver/Gold Parquet data, and
generates a business report. The repository includes the phased agent workflow,
Streamlit interface, shared state, audit logging, traces, and memory helpers.

## Install Dependencies

From the project root in Windows PowerShell, create a virtual environment and
install the dependencies without needing to activate it:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Requirements use version ranges, not a reproducible lockfile. Python version
support depends on the resolved packages and their available wheels.

## AI Providers

Claude is the primary/default provider through `langchain-anthropic`. The model
factory reads keys and model names from environment configuration; do not commit
secret keys.

Groq, Google, OpenAI and GitHub Models are supported as optional alternatives
(set `LLM_PROVIDER` and the matching key in `.env`); their credentials are not
needed for Claude-only use. The pipeline makes provider calls when the profile
meanings, the STTM mappings and the report are generated. Tests inject fake
models and do not make network calls.

## Environment Configuration

Copy `.env.example` to `.env` in the project root and fill in the key for only
your selected provider. Leave `LLM_PROVIDER=anthropic` for the recommended Claude
setup and provide `ANTHROPIC_API_KEY`. Local `.env` files are ignored by Git;
never put real keys in the example file.

The template pins `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`. Optional model
settings apply only to their selected providers. For Groq, the template uses
`openai/gpt-oss-20b`; code that parses model output should account for its
reasoning behavior. GitHub Models is also supported as an optional provider.

## Run the App

Copy `.env.example` to `.env`, add the key for your selected provider, and run
from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

If the key is missing the sidebar says so and the Start button stays disabled.

1. Upload one or more CSV files (each becomes a table named after the file) and
   type a business question, then press **Start pipeline**.
2. After each layer the run **pauses** on that layer's mapping (STTM). Edit any
   cell in the table (add or delete rows, change types and rules), then choose:
   - **Approve & continue**: your edits are validated against the real columns;
     problems are listed and nothing runs until they are fixed.
   - **Regenerate…**: describe what should change and the mapping is redesigned.
   - **Reject…**: stops the run.
3. After the Gold mapping is approved the report appears (charts, KPIs, the
   queries behind every number), with HTML and JSON downloads.

If a phase fails, the page shows the error with **Retry phase**. If the error
says the data conflicts with an approved mapping, use **Edit approved mapping**
to change the rules and approve again. Every run is saved under `data/runs`;
the browser address keeps `?run=<id>`, so a refresh or bookmark returns to the
same run, and **Resume a run** in the sidebar lists earlier ones. The **Agent
traces** and **Audit log** panels show what each agent did and every decision.

The profiler reads your data before any AI does and passes the AI what it found: date formats (including
`yyyymmdd` numbers), numbers stored as text, sign conventions, padded text, links between tables (found by
matching values, even when the columns are named differently), and months with only part of their days.
Columns that look like card numbers, bank details, ids, passwords or contact details are hidden from the AI
(values show as `[redacted]`) and a warning appears if a mapping keeps them. Each mapping review also shows
what the previous step did to your data (rows in and out, dropped, filled, unconvertible).

Data problems are fixed by rules in the mappings, which you can read and edit before approving: dates to real
dates (`to_date`), numbers stored as text (`clean_number`, `clean_number:eu`), padded or inconsistent text
(`trim`, `lowercase`, `map_values`), impossible values (`valid_range`), wrong signs (`abs`), nulls and
duplicates (`fillna`, `dropna`, `dedupe`). Values that cannot be converted become nulls and are counted, never
dropped silently.

Mapping generation rejects missing input columns, duplicate target columns,
unsupported operations, and aggregations outside Gold. CSV profiling rejects
ambiguous headers. Explicitly non-nullable targets fail if a conversion would
produce a missing value. Report charts and KPIs are drawn from SQL results the
Reporter actually ran; the narrative text is written by the model and should be
checked against the listed queries. A live run requires a valid provider key.

## Tests

Run the offline unit and end-to-end checks with:

```powershell
python -m unittest discover -v
```

The tests use temporary CSV/Parquet files and fake model responses; live AI
provider calls are not required.