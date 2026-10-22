# Claims Intelligence — Local Agent

Minimal setup steps to run and test locally.

Prerequisites:
- Python 3.10+
- PostgreSQL accessible with a `claims_db` database

Setup:

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure PostgreSQL credentials in a `.env` file or environment variables:

```
PGHOST=localhost
PGPORT=5432
PGDATABASE=claims_db
PGUSER=youruser
PGPASSWORD=yourpassword
PG_STATEMENT_TIMEOUT_MS=30000
```

4. Run a quick syntax check:

```bash
python -m compileall -q .
```

5. Run the database test suite (requires DB connection):

```bash
python tools.py
```

Or run specific checks in an interactive Python session:

```bash
python -c "from tools import database_health; print(database_health())"

## Run the applications

The primary user interface is `app.py`. It includes claims analytics,
natural-language database questions, visualizations, and investigation cases:

```bash
streamlit run app.py
```

The investigation API is a separate backend for external clients:

```bash
uvicorn investigation_api:app --reload
```

## Live database generator

`live_claim_generator.py` only produces claim payloads. It never connects to
PostgreSQL. The primary Streamlit app (`app.py`) starts the `agentfor.py`
ingestion loop automatically when writer credentials are configured. It inserts
an initial batch on startup, then generates another batch every 10 seconds by
default.

Configure separate credentials; do not put writer credentials in `PGUSER` or
`PGPASSWORD` because those variables are used by the read-only agent:

```text
PGUSER=agent_ro
PGPASSWORD=read_only_password
PG_WRITER_USER=claims_generator
PG_WRITER_PASSWORD=writer_password
```

To disable app auto-start, set `AUTO_START_CLAIM_GENERATOR=false`. Set
`CLAIM_GENERATION_INTERVAL_SECONDS` to change the interval. To run ingestion
separately from the Streamlit app, use:

```bash
.venv/bin/python agentfor.py
```

Do not run this standalone service at the same time as the Streamlit app, or
both processes will generate claims. The producer can also be used independently
by importing `generate_claim()` or `generate_claims()`. The ingestion service
does not delete existing records. Stop the standalone service with `Ctrl+C`.

## Anomaly analysis

The primary app exposes four deterministic analysis modes:

- Robust county investigation: MAD-based distribution and structural signals.
- Robust county anomaly ranking: two-part monetary features scored with median/MAD.
- Peer-normalized claim scoring: claim amount compared with type/facility-level peers.
- Temporal anomaly detection: county-month volume and exposure compared with prior months.
- Evidence report: monetary, insurer, facility, type, repetition, and daily-activity evidence.
- Legacy county profile: detailed county, national comparison, and distribution diagnostics.
- Database-wide complex analytics: volume, amount, monthly, cancellation, concentration, and data-quality findings.

Recommended algorithms for this claims data:

1. Use median/MAD or IQR and log-transformed amounts for robust outlier scores.
2. Use a two-part monetary model: zero-claim rate plus the positive-amount distribution.
3. Normalize claims against peers by claim type, facility, insurer, and facility category.
4. Use rolling temporal baselines; Poisson or negative-binomial models are better for claim counts than ordinary z-scores.
5. Add concentration and repetition features, including facility/insurer exposure share, repeated amounts, same-client same-day activity, and bundle/link relationships.
6. Use Isolation Forest only after creating a claim-level feature table. It should prioritize investigations, not estimate fraud probability.

The peer and temporal modes are deterministic and thresholded for review. They
do not label claims as fraudulent; they prioritize records or periods for
investigation.

### How the advanced detectors affect the system

- **Isolation Forest + LOF:** When selected in the app, the system loads a bounded recent claim sample and builds features from amount, cancellation, repeated amounts, same-client-day activity, same-facility-day activity, and client-facility repetition. Isolation Forest finds globally unusual combinations. LOF compares each claim with its nearest feature neighbors. Claims flagged by both models are shown as the strongest candidates.
- **Client-facility relationship anomalies:** The system groups claims into client/facility edges and ranks unusually dense relationships by interaction count. This surfaces relationships that need contextual review; it does not imply wrongdoing.
- **Poisson vs Negative Binomial check:** The system measures monthly claim-count dispersion. If variance is close to the mean, Poisson monitoring is recommended. If variance is much larger, Negative Binomial monitoring is recommended because the data is overdispersed.

The detectors are intentionally run only after the user submits an analysis
mode. They do not run during normal dashboard refreshes or ordinary natural-
language questions. Their outputs are stored in the Streamlit session and are
displayed as investigation-priority evidence alongside the existing case
workflow.

All anomaly outputs are investigation-priority evidence. They are not fraud determinations.
```

## Performance & LLM Optimization

The agent incorporates several key optimizations for low-latency, deterministic execution:

1. **Structured Output (`response_format={"type": "json_object"}`)**:
   - **Complexity Scoring**: Uses Groq/OpenAI JSON mode with `{"score": <1-10>}` to eliminate regex parsing errors and shave off unnecessary tokens.
   - **SQL Generation**: Constrains local Ollama generation to JSON format (`{"sql": "<PostgreSQL query>"}`), pruning the search space and speeding up token decoding.

2. **Complexity Cache (`_COMPLEXITY_CACHE`)**:
   - Scores are cached in memory keyed on `normalize_question()`, mirroring `_QUERY_CACHE`.
   - Repeated questions and template/button clicks avoid paying a remote Groq round-trip.

3. **Model Residency (`keep_alive`)**:
   - Configured via `OLLAMA_KEEP_ALIVE=30m` (or `-1` for indefinite).
   - Eliminates cold reload latency spikes between user queries on CPU and consumer GPU setups.

4. **Model Quantization (Q4_K_M GGUF)**:
   - Recommended quantization level: `Q4_K_M` (e.g. `qwen3.5:4b` or `llama3.2:3b`).
   - Offers the optimal balance between SQL accuracy, VRAM/RAM footprint, and CPU inference speed.

5. **Token Prediction Limit (`num_predict` / `MAX_SQL_TOKENS`)**:
   - Reduced from 1200 to 200 (`MAX_SQL_TOKENS=200`).
   - Slashes tail latency by preventing runaway generation and reducing backend output buffer allocation.

6. **Tightly Sized Context Window (`num_ctx`)**:
   - Measured against fixed `DATABASE_SCHEMA` + system prompt (~780 tokens).
   - Set to `NUM_CTX=2048` instead of oversized defaults (4k–131k), saving significant KV-cache RAM and compute.

7. **Complex-query review**:
   - Deterministic planner matches continue to execute their validated SQL directly.
   - Questions scored in the `MULTI_STEP` band (8–10) use the configured hosted API to propose SQL; the local Ollama model must explicitly approve it before execution.
   - Configure `COMPLEXITY_API_KEY` or `GROQ_API_KEY` to enable hosted scoring and SQL generation.
   - Set `REMOTE_SQL_MODEL`, `REMOTE_SQL_MAX_TOKENS` (default `1200`), and `REMOTE_SQL_TIMEOUT` to configure hosted SQL generation. The model defaults to `COMPLEXITY_MODEL` and the timeout to 30 seconds.
   - The system never executes model-generated Python. SQL remains read-only and is validated before it reaches PostgreSQL.

