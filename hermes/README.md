# Hermes Agent integration

This directory wires [Hermes Agent](https://github.com/nousresearch/hermes-agent)
(Nous Research's self-improving operator agent) into this trading bot as the
**reflection / self-improvement** layer.

Hermes runs as a **separate process** and drives the bot's nightly strategy
tuning *through a tool surface* — it never imports the trading code and never
touches the live signal path. The tool surface is:

    kronos_integrated_bot/reflection_cli.py   # JSON CLI over the reflection agent

and the agent-side instructions live in:

    hermes/skills/trading/kronos-reflection/SKILL.md

## How the pieces fit

```
Hermes Agent (own process, own LLM + persistent memory)
   │  runs terminal commands per SKILL.md
   ▼
python -m kronos_integrated_bot.reflection_cli <state|metrics|propose|apply|revert>
   │  reuses reflect.py — same guardrails as the in-house nightly cycle
   ▼
kronos_strategy.yaml  +  state/hypotheses.jsonl   (the bot reads these live)
```

Hermes decides *which* parameter to move and *to what*, using the same evidence
(`metrics`, `context`) the in-house DeepSeek proposer sees. But every write is
re-validated by the CLI through PARAM_BOUNDS clamp, the locked-param veto, the
evidence gate, the anti-oscillation guard, and replay validation. **Hermes
cannot bypass these** — a blocked write returns `{"ok": false, "refused": ...}`.

This means you can adopt Hermes incrementally: the CLI is useful and safe on its
own, and Hermes is just a smarter caller of it.

## Install the skill into Hermes

Hermes loads skills from `~/.hermes/skills/`. Copy (or symlink) this skill there:

```bash
mkdir -p ~/.hermes/skills/trading
cp -r hermes/skills/trading/kronos-reflection ~/.hermes/skills/trading/
# or symlink so repo edits propagate:
ln -s "$(pwd)/hermes/skills/trading/kronos-reflection" ~/.hermes/skills/trading/kronos-reflection
```

Then set the two config keys the skill expects (paths to the repo and its venv
python), e.g.:

```bash
hermes config set kronos.project_dir "C:/Users/Amit/Desktop/Deepseek-tradingbot"
hermes config set kronos.python "C:/Users/Amit/Desktop/Deepseek-tradingbot/.venv/Scripts/python.exe"
```

Verify Hermes sees it: `hermes` → `/skills` (or ask it "reflect on the bot").

## Schedule the nightly run (optional)

Use Hermes' built-in cron to run the loop after the NSE close (≈15:45 IST). In
the Hermes CLI:

```
hermes cron add --schedule "45 15 * * 1-5" \
  --prompt "Use the kronos-reflection skill: check the evidence gate; if open, review metrics and apply at most one bounded change; otherwise report the gate status."
```

(Confirm exact `hermes cron` flags against `hermes cron --help` for your version.)

## Windows / WSL note

The bot runs on Windows (`.venv\Scripts\python.exe`). Hermes' Windows installer
bundles its own runtime; if you instead run Hermes under WSL2, have it call the
Windows interpreter across the boundary, e.g.:

```bash
/mnt/c/Users/Amit/Desktop/Deepseek-tradingbot/.venv/Scripts/python.exe \
    -m kronos_integrated_bot.reflection_cli state
```

(run with the repo root as CWD). Set `kronos.python` / `kronos.project_dir`
accordingly for whichever side Hermes runs on.

## CLI reference (also runnable without Hermes)

| Command | Effect |
|---|---|
| `reflection_cli state` | strategy version, tunable params + bounds/flags, evidence-gate status |
| `reflection_cli metrics` | performance since last change: aggregate + by regime/direction + indicator buckets |
| `reflection_cli hypotheses -n N` | last N reflection decisions |
| `reflection_cli context` | the exact system+user prompt the in-house proposer would get |
| `reflection_cli propose P V [--replay]` | DRY validation of moving param `P` to `V` (no write) |
| `reflection_cli apply P V --reason "..." --confirm` | validate AND write (gated; refused if gate closed/locked/oscillation/replay-degraded) |
| `reflection_cli revert [--confirm]` | auto-revert the last change if it degraded |
| `reflection_cli schema` | read-only schema + per-column fill rates of `analog_history.db` |
| `reflection_cli query "SELECT ..." [--limit N]` | run a single **read-only** SELECT/WITH against `analog_history.db` (for structural diagnosis) |
| `reflection_cli run [--dry-run]` | the full in-house autonomous cycle (DeepSeek picks the move) |

### Read-only safety of `query`

`query` is constrained three independent ways so it can never mutate the
ground-truth trade DB: the connection is opened `mode=ro`, an authorizer denies
every action except SELECT/READ/FUNCTION, and the SQL must be a single
SELECT/WITH statement (DELETE/UPDATE/DROP/INSERT/PRAGMA and multi-statement
inputs are refused). Results are capped at `--limit` rows (default 100, max
2000) with a `truncated` flag.
