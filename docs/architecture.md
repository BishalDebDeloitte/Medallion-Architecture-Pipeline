# Pipeline Architecture and Handoff Contract

## Control boundaries

- `agents/orchestrator.py`: deterministic run initialization, phase sequencing, human
  approval gates, structured error handling, audit events, and copying declared scratchpad keys
  into `PipelineState`. Never calls an LLM (see "Orchestrator" below).
- `agents/supervisor.py`: LLM ReAct interface with only the supplied phase tools;
  it does not own or mutate `PipelineState` or Streamlit state.
- `core/scratchpad.py`: typed in-run artifacts written by tools and read by the
  Orchestrator. State changes occur only after successful handoff validation.

## Handoffs

All paths are run-scoped (`{run_id}` is a UUID) so runs never overwrite each other.

| Phase | Sender to receiver | Explicit inputs | Scratchpad output | Approved handoff |
| --- | --- | --- | --- | --- |
| 0 | User to Orchestrator | CSV paths, business intent | none | `run_id`, initial state |
| 1 | Supervisor to Profiler | CSV paths, run ID, task description | `profile_path`: `data/profiles/profile_combined_{run_id}.json` | STTM context |
| 1 | Supervisor to STTM (Bronze) | profile path, intent, run ID | `sttm_bronze_path`: `data/sttm/sttm_bronze_{run_id}.csv` | human approval required |
| 2 | Supervisor to Bronze | CSV paths, approved Bronze STTM, run ID | `bronze_output_paths`: `data/bronze/{source_stem}_{run_id}_bronze.parquet` | Silver STTM context |
| 2 | Supervisor to STTM (Silver) | Bronze paths, Bronze STTM, run ID | `sttm_silver_path`: `data/sttm/sttm_silver_{run_id}.csv` | human approval required |
| 3 | Supervisor to Silver | Bronze paths, approved Silver STTM, run ID | `silver_output_paths`: `data/silver/{source_stem}_{run_id}_silver.parquet` | Gold STTM context |
| 3 | Supervisor to STTM (Gold) | Silver paths, Silver STTM, run ID | `sttm_gold_path`: `data/sttm/sttm_gold_{run_id}.csv` | human approval required |
| 4 | Supervisor to Gold | Silver paths, approved Gold STTM, run ID | `gold_output_paths`: `data/gold/{target_table}_{run_id}.parquet` | Reporter input |
| 4 | Supervisor to Reporter | Gold paths, intent, run ID | `report_path`: `data/reports/report_{run_id}.html`; `report_json_path`: `data/reports/report_{run_id}.json` | UI input |
| Final | Orchestrator to Streamlit | report path | HTML report | displayed to user |

Directory constants and artifact patterns live in `core/paths.py`. Set the
`MEDALLION_RUNTIME_DIR` environment variable to move all runtime output (data,
audit logs, memory); the test suite does this so tests never write to `data/`.

## Orchestrator (agents/orchestrator.py) and the approval gate

Deterministic and LLM-free. `start_run` creates the run and executes Phase 1; `run_next_phase`
derives the next phase only from which artifacts exist, builds the phase's tools
(`agents/toolkit.py`), and hands the Supervisor a fixed goal. After the phase it validates the
scratchpad and only then copies outputs into `PipelineState`. STTM outputs go to
`pending_approval_path`, never to `sttm_<layer>_path`, and the run waits in `awaiting_approval`.

- `approve(state, rows=None)` re-validates the (possibly human-edited) rows against the real source
  columns, writes `sttm_<layer>_<run>_approved.csv` (the generated file is kept) and only then sets
  `sttm_<layer>_path`. Invalid edits raise and change nothing.
- `regenerate(state, feedback)` re-runs only the STTM agent; on failure the previous mapping stays
  reviewable. `cancel` is terminal. `retry` re-runs a failed phase. `reopen_mapping` puts an
  approved mapping back in review when it failed on the data (a retry would fail the same way).
- State is saved to `data/runs/<run_id>.json` after every transition; `resume(run_id)` reloads it
  (a run found `running` after a restart is marked failed as interrupted, so it can be retried).
- Every decision is audited with its rationale (`phase_selected`, `sttm_approved` with `edited`,
  `regenerate_requested`, `run_cancelled`, ...).

## Streamlit app (app/streamlit_app.py, app/ui_logic.py)

The app holds an `Orchestrator` and the current `PipelineState` in `st.session_state` and keeps
`?run=<id>` in the URL. Pure logic (tracker, editor/STTM conversion, upload validation) is in
`ui_logic.py` so it is unit tested. Uploads are saved as `data/landing/<batch>/<name>.csv`. All
actions run inside a spinner with errors shown, never raised; invalid reviewer edits are shown above
the decision buttons and the editor keeps them.

## Run status

`PipelineState.status` is one of `pending`, `running`, `awaiting_approval`,
`complete`, `failed`. While `awaiting_approval`, `pending_approval_path` and
`pending_approval_layer` identify the STTM under review. `PipelineState` validates
the status, so an unknown value fails immediately.

## STTM contract (`core/sttm_schema.py`)

One CSV per layer and run, one row per source column to target column:

`layer, target_table, source_table, source_col, target_col, data_type,
transformation_logic, aggregation, nullable, join_key, description`

- `transformation_logic` is a pipe-separated list of allowlisted operations, never code:
  - text: `identity, trim, lowercase, uppercase` (Bronze too), `titlecase, collapse_whitespace`;
  - types: `cast:<type>`, `to_date[:strptime format]` (for example `to_date:%d/%m/%Y`, or
    `to_date:%Y%m%d` for 20250402), `date_part:<year|quarter|month|week|day|weekday|year_month|
    year_quarter|month_name|day_name>`, `clean_number` ("₹1,234.50", "USD 1,200" or "(12)" to a number;
    `clean_number:eu` for "1.234,56"; text that is still not a number afterwards becomes a counted null),
    `abs`, `round:<0-6>`;
  - data quality (Silver and Gold): `map_values:<value>=<replacement>;...` (exact replacements to standardise
    categories such as "n.y." to "ny", or `unknown=` to turn a placeholder into null; values cannot contain
    `| ; =`), `valid_range:<min>..<max>` (either side optional; values outside, such as age 250 or a negative
    quantity, become counted nulls that `dropna` or `fillna` can then handle);
  - cleansing (Silver only): `dropna`, `fillna:<mean|median|mode|zero|empty|value=X>`, `dedupe`.
  Chains run left to right (`to_date:%d/%m/%Y|date_part:year_month`); the data_type must match what
  the chain produces (`date_part:year` is integer, `year_month` is string, `abs` is numeric).
- `aggregation` (`sum, avg, mean, count, max, min, first, last, nunique`) is
  Gold-only; non-aggregated Gold columns are the group-by keys.
- `join_key` (Gold): the shared `_id` column used to outer-merge the
  `source_table`s of a `target_table`. Required when a Gold table has more than
  one source table.
- Validation (`validate_rows`) checks layer, duplicate targets, allowlists and,
  when given the source schema, that every source table/column exists.

## Specialist agents

Each specialist is a ReAct agent over its own tools (`core/agent_runner.py`), exposed to the
Supervisor as one tool by a `make_*_agent_tool` factory, and records an `AgentTrace`.
Tools return JSON and never raise (`core/tooling.py`); execute tools refuse to run before
the inspect tool (`InspectGate`); agents may only touch paths they were given.

- **Profiler** (`agents/profiler.py`): `inspect_files_tool` then `profiler_tool`. The LLM supplies
  `semantic_meanings`; statistics, join keys and quality notes are computed in pandas.
- **STTM** (`agents/sttm_generator.py`): one agent per layer with only `inspect_context_tool`
  and that layer's `generate_<layer>_sttm_tool`. The LLM authors the rows; the tool validates
  them against the real source columns (all problems reported at once, nothing written on
  error) and writes the CSV. Bronze reads the profile; Silver reads Bronze Parquet; Gold reads
  Silver Parquet. To regenerate after a rejection, rerun with the reviewer feedback in the
  `task_description`; the run-scoped STTM file is overwritten.
- **Bronze / Silver** (`agents/bronze_agent.py`, `agents/silver_agent.py`): `inspect_task_tool`
  (data preview plus the approved STTM rules and a validity check) then `bronze_ingestion_tool` /
  `silver_ingestion_tool`. They execute the approved STTM and never change it: the LLM plans,
  verifies the rules look valid and reports; execution is deterministic (`core/transforms.py`).
  An execution error is final and is reported, not retried. The tools are built only when
  `approved=True`. All tables are transformed and checked before any file is written, and the
  outputs are verified to contain exactly the STTM-approved columns (Silver: `pk_<table>_silver_id`
  first). Bronze reads CSVs as text, keeps dates as raw strings and applies only
  trim/lowercase/uppercase and the data_type cast. Silver applies, in order: text/date operations
  and cast, `fillna`, `dropna`, `dedupe` (only on columns marked `dedupe`), then the
  `nullable=false` check. Unconvertible values become null and are counted in the tool result;
  dates are parsed month-first unless `to_date:<format>` says otherwise.
- **Gold** (`agents/gold_agent.py`): same inspect-then-execute shape. Per target table it
  outer-merges the Silver sources on the STTM `join_key` column(s), applies column operations,
  groups by the non-aggregated columns and aggregates the measures, casts to `data_type`,
  enforces `nullable`, and injects `pk_<table>_gold_id` as column 0. Safeguards: rows with a null
  key never match each other (they are kept as separate rows); a join where the key repeats on
  both sides is refused because it would multiply rows and inflate every sum; key types must be
  compatible; `sum`/`avg` need a numeric source; measures are aggregated before their final cast;
  a group with only null inputs yields null (not 0) for `sum`, and `count` yields 0. The inspect
  tool dry-runs every join, so an unsafe STTM is reported before anything runs. A target table
  with no aggregation is a distinct-rows dimension table; one with only aggregations is a single
  row of totals.
- **Reporter** (`agents/reporter.py`, `core/report_spec.py`, `core/report_render.py`): three tools,
  `inspect_gold_tables_tool` (Parquet metadata + 3 sample rows), `load_gold_data_tool` (registers the
  Gold files as in-memory DuckDB tables and returns the exact catalog, with column descriptions when
  the Gold STTM is supplied) and `execute_query_tool(sql)`. The agent ends with a JSON report spec
  (headline, summary, KPIs, insights, charts, limitations). Code validates it against the queries that
  actually ran, then renders the report, so charts and KPIs are drawn from stored query results, never
  from model-typed numbers; the narrative text is the model's and the report says so.
  - *Sandbox:* one read-only `SELECT`/`WITH` per call; external file access, extension loading and
    configuration changes are disabled before the first query; results are capped (20,000 rows), as are
    queries (15), runtime (30 s) and memory (1 GB). Two SQL errors in a row stop querying (the original
    attempt plus one retry).
  - *Correction loop:* an unusable final answer (bad JSON, unknown query or column, a chart-rule
    violation) is sent back to the agent with the specific problems, up to twice.
  - *Chart rules (enforced in `validate_spec`):* no one-bar charts or 2-slice pies (use KPIs), pies need
    3-6 slices, at most 30 bars, series capped at 6 (3 for scatter) and 8 colour-coded entities per
    report. Rendering: one colour for a single series; the validated categorical palette in fixed order
    for several, assigned per entity across the whole report; thin marks with a 2px surface gap;
    hairline solid grid; a legend whenever there are 2+ series; hover tooltips; a collapsible data table
    for every chart; horizontal bars for long labels. The report is light-themed and printable, escapes
    all model and data text, and works at phone width.
  - *Outputs:* `data/reports/report_{run_id}.html` (Plotly JS embedded so it opens offline; set
    `REPORT_PLOTLY_JS=cdn` for a ~100 KB file that needs internet) and `report_{run_id}.json` (content,
    figures, queries, colour assignments). The intent and a plain-text report are stored with
    `store_document()`.
- **Memory** (`core/memory.py`): ChromaDB with a constant placeholder embedding per document, so the
  default ~80 MB embedding model is never downloaded; retrieval is by id or in batches, not semantic.

## Understanding the data (core/profiling.py) and protecting sensitive data (core/sensitivity.py)

The profile is what the STTM agent designs transformations from, so it carries deterministic signals:

- **Types and formats:** dates stored as ISO text, `dd/mm/yyyy` or `mm/dd/yyyy` text (flagged `ambiguous`
  when every day and month value is 12 or less) or `yyyymmdd` integers in a date-named column; row-number
  columns (`Index`, `Row`, 1..n) are recognised and never used as keys.
- **Hints:** per column, operations the data calls for (`trim`, `lowercase`, `to_date:<format>`,
  `clean_number` or `clean_number:eu`, `abs` for an all-negative amount, `dedupe` for a key with duplicates,
  `trim|lowercase|map_values:...` for spelling and punctuation variants of one category, `cast:integer|float`
  for a column that is numeric apart from a few placeholders, `cast:boolean` for yes/no text, and
  `valid_range:0..120` or `valid_range:0..` for impossible ages and negative quantities or prices). Hints are
  suggestions; the STTM agent chooses and the reviewer approves. Impossible values also appear in the
  quality notes.
- **Time coverage:** for dense date columns, partial first/last months ("2026-04 has 2 of 30 days").
  Birth-date columns are skipped.
- **Join discovery by value, not by name:** a unique column whose values contain nearly all of another
  column's values is a link, whatever the columns are called. Dates are compared as dates across formats.
  Same-named keys may have orphans (>= 50% match); differently named columns need >= 95% and either related
  names (`cust_ref` -> `customer_id`) or a large exact text match, and the "many" side must repeat (two
  sequential id columns always contain each other). One link is kept per column and table. Each link carries
  `alignment` steps (rename and convert both sides in Silver so Gold can join on one shared column).
- **Sensitive columns:** names (card, cvv, iban, ifsc, upi, account number, ssn, aadhaar, passport, password,
  token, email, phone, ...) and values (Luhn-valid card numbers, emails) mark a column financial, identity,
  credential or contact. Their sample values, statistics and hints are replaced by `[redacted: <kind>]`
  in the profile, in every preview, and in SQL results shown to the Reporter. The pipeline still processes
  the real values if a mapping keeps them; the STTM tool warns, and the review screen shows a warning.

## What each step reports

Bronze, Silver and Gold write `data/layer_reports/<run>_<layer>_report.json` (rows in and out, rows dropped
by `dropna` and `dedupe`, values filled, values that could not be converted, join keys and merged rows). The
review screen shows the previous step's report ("What the last step did") before you approve the next
mapping.

## Partial periods in reports

A chart that shows a month with only part of its data would mislead (a cliff on the last point). The Reporter
catalog lists the partial months of each Gold table, tied to the date column the table was built from (via
the Gold mapping's lineage and the raw-data profile), and the prompt tells the agent to exclude them. As a
safety net, `report_render` adds a visible warning under any chart whose x values include one
("2026-04 covers only 2 of 30 days, so it is not comparable with full periods"), and records it in the JSON.

## Project layout

```text
medallion-pipeline/
  agents/              orchestrator.py, supervisor.py, toolkit.py (wires the real
                       agent tools per phase), and the specialist agents
  app/                 streamlit_app.py (UI) and ui_logic.py (testable UI logic)
  core/                paths.py, state.py, scratchpad.py, run_store.py, sttm_schema.py,
                       transforms.py, csv_io.py, tooling.py, agent_runner.py,
                       layer_agent.py, report_spec.py, report_render.py, config.py,
                       llm.py, observability.py, audit.py, memory.py
  data/landing/        uploaded CSV files (one folder per upload batch)
  data/runs/           saved run state (resume after a refresh)
  data/profiles/       raw and layer profiles
  data/sttm/           proposed and approved mappings
  data/bronze/         Bronze Parquet
  data/silver/         Silver Parquet
  data/gold/           Gold Parquet
  data/reports/        HTML and JSON executive reports
  data/traces/         agent traces
  audit_logs/          run-level audit JSONL
  tests/               offline interface and prior workflow tests
```
