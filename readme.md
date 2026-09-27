# Vera AI Challenge — Submission

**Team:** Arnav Sharma · **Model:** Gemini `gemini-3.1-flash-lite` (temperature=0)

## Approach

A router + validator pipeline, not a single mega-prompt. `/v1/tick` composes each
active trigger via Gemini, grounded in that trigger's own payload plus category
voice and merchant context. If Gemini fails, times out, or returns something
malformed, a **trigger-aware deterministic template** takes over — every trigger
kind has an explicit template branch pulling facts straight from `trigger.payload`,
so a failure never means a generic message, it means a *specific* one composed
without an LLM.

## Composition logic

- **LLM path** (`_llm_compose`): builds a JSON context (category voice/taboos,
  merchant identity + performance + active offers + signals, trigger kind/scope/
  payload, matched digest item, customer data if present) and asks Gemini for
  `{body, cta, rationale}`.
- **Guardrail before the LLM path is trusted** (`_validate_llm_result`): rejects
  empty bodies, bodies >700 chars, an unrecognized/missing CTA, or any URL —
  falls back to the template on any violation. Prompt rules aren't enforcement;
  this is.
- **Customer-scoped triggers without customer data** (e.g. `recall_due` when no
  customer context was pushed) skip the LLM entirely and go straight to the
  template. Gemini repeatedly confused "message about a patient" with "message
  about the merchant themselves" when it had no patient object to ground on —
  the template doesn't have this failure mode, so we don't ask the LLM to guess.
- **Template fallback** never defaults to CTR/performance numbers for a trigger
  kind it doesn't specifically recognize — a wrong metric is worse than an
  honest, vague line.

## Reliability / guardrails

- **Concurrent composition, rate-limited**: all triggers in a `/v1/tick` batch
  compose in parallel (`asyncio.gather` + `asyncio.to_thread`, since the Gemini
  call is blocking I/O) capped at 5 simultaneous calls via a semaphore — avoids
  both the sequential-timeout problem (N triggers × up to 8s each) and
  self-inflicted 429s from bursting the API.
- **Two-phase suppression commit**: a trigger's `suppression_key` is only
  permanently marked "sent" after an action is actually built and returned. If
  composition throws mid-flight, the reservation is released so the trigger can
  be retried on the next tick instead of being silently dropped forever.
- **`/v1/context` idempotency**: versioned per `(scope, context_id)`, returns
  `409 stale_version` on a repeat/older version, validates `version` is a
  non-negative int before comparing (malformed input is rejected cleanly, not a
  500).
- **`/v1/reply`**: auto-reply ladder (send once → wait 24h → end) with a
  word-boundary negative-intent check (won't false-positive on "know" or "no
  problem"), a sticky `ended` state so a hard decline never re-engages, and
  anti-repetition suffixing on repeated static bodies.

## Known tradeoffs

- **In-memory state** (`STORE`, `CONVERSATIONS`, suppression sets) — intentional
  for the challenge window; a production deployment would externalize this to
  Redis/Postgres so state survives a restart.
- **CTA selection for trigger kinds beyond the documented examples** is
  heuristic (slot-based → `multi_choice_slot`, info-only → `open_ended`,
  everything else → `binary_yes_no`).

## Deploy

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # or a .env file (python-dotenv is loaded)
uvicorn main:app --host 0.0.0.0 --port 8080
```

Deployed on Render, connected to this GitHub repo with auto-deploy on push.
`GEMINI_API_KEY` set as a Render environment variable (not committed).