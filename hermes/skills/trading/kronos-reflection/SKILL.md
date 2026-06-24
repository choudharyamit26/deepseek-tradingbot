---
name: kronos-reflection
description: Drive the nightly self-improvement loop of the Kronos intraday trading bot — read trade performance, propose one bounded strategy-parameter change, and apply it through the bot's own guardrails. Use after market close or when asked to review/tune the strategy.
version: 1.0.0
metadata:
  hermes:
    tags: [trading, reflection, self-improvement, nse]
    category: trading
    requires_toolsets: [terminal]
    config:
      - key: kronos.project_dir
        description: "Absolute path to the Deepseek-tradingbot repo root"
        default: "C:/Users/Amit/Desktop/Deepseek-tradingbot"
        prompt: "Path to the trading-bot repo root?"
      - key: kronos.python
        description: "Python interpreter that has the bot's deps (the repo venv)"
        default: "C:/Users/Amit/Desktop/Deepseek-tradingbot/.venv/Scripts/python.exe"
        prompt: "Path to the bot's venv python?"
---
# Kronos Bot — Reflection / Self-Improvement

Operate the trading bot's nightly tuning loop **as a set of tools**, with deeper
memory and cross-session context than the bot's built-in single-shot proposer.
You are the strategist; the bot's CLI is the authoritative gatekeeper.

All commands go through one JSON CLI. Substitute the configured paths:

```
PY="<kronos.python>"            # e.g. .../.venv/Scripts/python.exe
cd "<kronos.project_dir>"
$PY -m kronos_integrated_bot.reflection_cli <subcommand> ...
```

Every call prints **one JSON object to stdout**. Parse it; never scrape logs.

## When to Use
- Daily after the NSE close (≈15:45 IST) to review the session and tune.
- When the user asks to "reflect", "review the strategy", "why are we losing on
  X", or "should we change a parameter".
- **Not** during market hours for per-trade decisions — this is the strategy
  layer, not the signal path.

## The one rule you cannot break
The CLI enforces the guardrails; **you do not get to override them**:
- `apply` is **refused** when the evidence gate is closed (fewer than
  `min_trades_per_change` closed trades since the last change). A closed gate is
  the system working correctly — **do not** try to work around it, lower the
  goal, or edit `kronos_strategy.yaml` by hand. Report "gate closed, N/M trades,
  waiting for more outcomes" and stop.
- Locked params (`risk_per_trade_pct`, `max_daily_loss_pct`, `max_daily_trades`,
  `max_consecutive_losses`) are capital-protection knobs and will be rejected.
- Proposals that merely undo the last change (oscillation) or that degrade the
  replay are refused. Pick a *different* parameter grounded in the evidence.
- Change **exactly one** parameter per session.

## Procedure
1. **Read state** — is the gate open?
   ```
   $PY -m kronos_integrated_bot.reflection_cli state
   ```
   If `evidence_gate.open` is `false`: report the count and **stop** (no change).

2. **Read the evidence** (only if the gate is open):
   ```
   $PY -m kronos_integrated_bot.reflection_cli metrics
   $PY -m kronos_integrated_bot.reflection_cli hypotheses -n 12
   ```
   Study `by_direction`, `by_regime`, and `indicator_analysis` to find the
   *specific losing bucket* (an RSI zone, ADX band, direction, volume band), and
   the change history to avoid repeating a move that already failed. Optionally
   pull the in-house proposer's exact prompt for full context:
   `... reflection_cli context`.

3. **Pick ONE parameter** that gates the losing bucket. Prefer filter-level gates
   (adx/rsi/volume/rr) over `replay_blind` params (`min_confidence`, `kronos_*`),
   which the replay cannot validate. Avoid any param flagged `recently_adjusted`.

4. **Dry-validate** the candidate (fast, no replay):
   ```
   $PY -m kronos_integrated_bot.reflection_cli propose <param> <value>
   ```
   Check `valid`, `was_clamped` (the CLI clamps to bounds/step — respect the
   clamped value), and `oscillation`. Iterate cheaply here until you have a
   clean candidate. Add `--replay` to preview the slow replay verdict.

5. **Apply** through every gate (this runs the slow replay; expect 30–120s):
   ```
   $PY -m kronos_integrated_bot.reflection_cli apply <param> <value> \
       --reason "<one-sentence evidence-grounded hypothesis>" --confirm
   ```
   Without `--confirm` it returns a dry preview (`would_apply: true`). With
   `--confirm` it writes, bumps the strategy version, archives the prior
   version, appends a hypothesis record, and fires the bot's Telegram alert.

6. **Revert check** (optional, run first on later nights) — auto-reverts the last
   change if it has enough evidence and clearly degraded:
   ```
   $PY -m kronos_integrated_bot.reflection_cli revert            # preview
   $PY -m kronos_integrated_bot.reflection_cli revert --confirm  # write
   ```

## Diagnosing structural problems (beyond parameter tuning)
Parameter tuning is local optimization — it cannot fix a strategy-design problem
(e.g. entering too late, a dead direction). To investigate the *root cause*, query
the real fill/PnL ground truth directly. This is **read-only** — you cannot
mutate `analog_history.db`; write attempts are refused.

1. Learn the shape first (tables, columns, and which columns are actually
   populated — ignore any with low `fill_pct`):
   ```
   $PY -m kronos_integrated_bot.reflection_cli schema
   ```
2. Ask diagnostic questions with a single SELECT. Examples:
   ```
   # Time-of-day decay (is the edge gone after the open?)
   $PY -m kronos_integrated_bot.reflection_cli query \
     "SELECT substr(ts,12,2) hour, COUNT(*) n, \
      ROUND(AVG(CASE WHEN outcome='WIN' THEN 100.0 ELSE 0 END),1) wr, \
      ROUND(SUM(pnl),2) pnl FROM setups GROUP BY hour ORDER BY hour"

   # Direction x Nifty-trend (where is the real edge / the leak?)
   $PY -m kronos_integrated_bot.reflection_cli query \
     "SELECT signal_type, nifty_trend, COUNT(*) n, ROUND(SUM(pnl),2) pnl \
      FROM setups GROUP BY signal_type, nifty_trend ORDER BY pnl"

   # Does going against the latest candle hurt?
   $PY -m kronos_integrated_bot.reflection_cli query \
     "SELECT candle_against, COUNT(*) n, ROUND(SUM(pnl),2) pnl FROM setups \
      GROUP BY candle_against"
   ```
3. Turn a finding into action the right way:
   - If it maps to an existing gate (a losing RSI/ADX/volume/time bucket), make
     a bounded `apply` (when the evidence gate is open).
   - If it's structural (late entry, a missing filter, a dead direction), it is
     **not** a parameter — write up the finding with the supporting query and
     numbers and surface it to the user as a proposed code/strategy change. Do
     not invent a new parameter or edit the bot's code from this skill.

## Pitfalls
- **stdout is JSON, stderr is logs.** If a command seems to print nothing, it is
  still running (the replay in step 5 is slow) — wait, don't re-run.
- A refusal (`"ok": false, "refused": ...`) is an *answer*, not an error to
  retry. Read `refused` and adapt.
- Don't propose a value outside `[min, max]` or more than `max_step` from
  current — the CLI will clamp it and the effective change may be smaller (or a
  no-op, which is rejected). Read `clamped_value` back.
- The bot's own `run` cycle (`... reflection_cli run`) uses DeepSeek to pick a
  move. Use it as a fallback/cross-check, not in combination with a manual
  `apply` in the same session (one change per cycle).

## Verification
After an `apply --confirm`, confirm the write landed:
```
$PY -m kronos_integrated_bot.reflection_cli state
```
The `version` should have incremented (e.g. `kronos-v19` → `kronos-v20`) and the
changed param should show the new `current` value. The latest `hypotheses -n 1`
record should describe your change with your `--reason`.
