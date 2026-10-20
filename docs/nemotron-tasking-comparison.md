# Which Nemotron for the tasking step, and why that size

The ground-side step turns a free-text tasking ("check whether the road under the slope is
still passable") into the structured observation goal the on-board model receives. The model
may only choose from the closed vocabulary of the mission profile catalog, so the task is
strict schema conformance rather than open-ended reasoning: a goal that is one field short
is useless to the compiler downstream.

## Method

16 cases, frozen in `training/tasking-goal-cases-v1.json` (sha256 `585ccb1c22aaeda26e33fe2a74c0f6e0e89cd44c9bdd11ebd75fb5744580f3fd`) **before any model
was called**. One call per case, `temperature=0`, `response_format=json_object`. Scored on
exact mission profile, exact category and F1 over the required questions. `map_relations` is
recorded but not scored, because more than one relation can be defensible.

Reproduce with `training/evaluate_tasking_models.py`; per-case outputs are in `results/`.

## Result (2026-10-21, Nebius Token Factory)

| model | schema valid | profile exact | category exact | question F1 | all three exact | median latency |
|---|---|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 100% | 100% | 100% | 0.958 | 14/16 | 2.41 s |
| `nvidia/Nemotron-3_5-Lightning` | 100% | 100% | 100% | 0.979 | 15/16 | 6.53 s |
| `nvidia/nemotron-3-super-120b-a12b` | 100% | 100% | 94% | 1.000 | 15/16 | 4.28 s |

**With 16 cases, 15/16 and 14/16 are not meaningfully different.** What the measurement does
support is narrower and more useful: the 30B-A3B class is enough for this step, and the
120B model buys nothing here. We run Lightning, and Nano is the option when latency matters
more than the last case.

## One trap worth passing on

Our first run scored 0.0 on every metric for all three models, which looked like a catalog
or prompt failure. It was neither. Nemotron 3.5 reasons by default and spends `max_tokens`
on reasoning tokens first, so a 400-token cap returned `content: null` with
`finish_reason: length`; a typical case here uses about 1,600 reasoning tokens. Raising the
cap to 3,000 produced correct answers from every model. **A run where unrelated metrics all
read zero is a failed invocation, not a finding.**
