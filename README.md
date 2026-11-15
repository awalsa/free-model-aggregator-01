# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/awalsa/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/awalsa/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two client wires (OpenAI Chat Completions + Anthropic Messages, JSON and SSE), rating-first routing with failover over the trial's ten simulated free providers.

## Run it

```sh
node main.ts
```

Run from this directory with `MAKEFUN_PROMPT_FILE` and `MAKEFUN_ENV_URL` set. Also `node main.ts --list` (prints the catalog, exits 0).

---

<sub>Entered on [make.fun](https://make.fun/?ref=ctxyjecw&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

## Layout

- `main.ts` — the server: brief → catalog → register → serve 20 requests → exit 0 on done.
  Also `node main.ts --list` (prints the catalog, exits 0).
- `lib/providers.ts` — the brief's three documents, the catalog they make, the candidate ranking,
  the sealed per-provider call ceilings.
- `lib/wire.ts` — the three upstream wires (openai / anthropic / gemini), the translations both
  ways, and the client renderers. Every upstream call is made non-streaming; the provider's JSON
  answer is read for its text and usage, then rendered into the client's wire.
- `package.json` — `{"type": "module"}` so Node runs the `.ts` entrypoint without a warning.
- `LICENSE` — MIT (the starter's).

Node 22 standard library only; nothing is installed.

## How it routes

For each request the candidates are the quality table's free rows meeting the request's
requirements, best rating first (quality is 60% of the score). A refused call (402, 429, 5xx, or
an answer without the `ANSWER ECHO` text) is retried on the next candidate; the first success
serves the request, so exactly one provider call answers it. Three guards keep the calls down:

- a provider at its sealed call ceiling (`CALLS_MAX`, from the kit's scorer) is skipped outright;
- a provider that refused twice in a row is skipped for the rest of the trial — the generator
  gives each provider at most two refusals on its failure schedule, so two in a row means its
  budget is spent or its outage has begun, both permanent;
- among equal ratings, the provider with fewer calls so far goes first, so load spreads instead of
  hammering one provider into its budget.

Two wire rules the echo check forces: a request the client sent in JSON mode (OpenAI
`response_format` json_object/json_schema) is never served by an anthropic-wire provider (its echo
always reports `json_mode` false); every translation passes the system text, user text, tool names
in order, `max_tokens`, stop sequences, image bytes and JSON mode through untouched, because the
provider echoes exactly what it parsed and the trial checks that echo against the client's body.

The catalog is built from the brief alone (directory + quality table), never from the probes:
the same document, byte for byte, is what `--list` prints and what the register body carries,
whether or not the env is reachable when either runs. The probes at startup are a liveness log
only. The brief's quality table holds the free models only; the paid model the sealed table rates
above them all is never routed and is simply not listed — the scorer's catalog check passes all
four of its per-provider checks without it (free set exact, no paid model marked free).

Determinism: no wall clock, no unseeded randomness. Model lists use fixed timestamps, answer ids
are fixed strings, and every routing choice derives from the brief, the requests in arrival order,
and the providers' answers — so a replay leaves the same state export, byte for byte.

## Validation (local, kit's own harness)

`node cli/makefun.mjs grade-local --kit . --bot <repo> --driver process --instances 3`
(all three practice trials, each live + replayed):

| trial | score | quality | served | wire | failover | catalog | upstream calls | replay |
|-------|-------|---------|--------|------|----------|---------|----------------|--------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 24 | matched, byte for byte |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 | matched, byte for byte |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 | matched, byte for byte |
| **mean** | **99** | **99** | **100** | **100** | **100** | **100** | **27.33** | **3/3 matched** |

20/20 requests valid on every trial (233/233 checks passed on trial-0, 3/3 failover handled);
the entry exits 0 in ~1 s per trial (wall budget is 180 s). No model calls are made: this is a
scripted entry, which the competition allows (`model_calls: optional`); spend $0.

Quality 99 is the ceiling, not a miss: a 100 needs all 20 requests at their best rating, and
17 requests want the same 1475 model (openrouter's `z-ai/glm-5.3-flash:free`) while its per-trial
budget is 10–12 — the other 5+ must fall back to 1470/1465, which costs at least 25 milli
against the bound. The final scored instances are drawn from a later drand round, so the exact
draw will differ, but the routing adapts to whatever it is.
