# Anvil

**Local-model iterative problem-solving harness with an immune system.**

Anvil wraps a small local LLM in an iteration loop instrumented by [anneal-memory](https://github.com/phillipclapham/anneal-memory)'s **four-layer memory**: an episodic store for raw attempts, a continuity file for compressed graduated patterns, Hebbian associations that strengthen cross-episode links under semantic judgment, and an affective limbic layer that weights what matters during consolidation. The model generates. The harness consolidates, validates, and decays. The immune system prevents the mem0-style context pollution that kills every other sustained-iteration loop.

In one sentence: **anneal-memory turns a local ~4B model into a patient, self-correcting problem solver that doesn't forget what it tried, doesn't repeat failed approaches, and consolidates insights as it goes.**

## Status

**Session 0 complete.** Baseline pipeline proven end-to-end on one bug across two local model families (Gemma 4 E4B, Qwen3.5-9B). First empirical finding already captured — see `results/bug_01_baseline.md` for the forensic.

Roadmap sessions 1-5 will generalize to 5 bugs, add the iteration loop, integrate anneal-memory, produce cross-family replication data for the memory-without-grounding thesis.

Early days. Not yet useful as a tool. Useful as a direction and as an empirical platform.

## Architecture

```
Problem in (description + failing test)
  │
  └─ Iteration loop:
       1. ATTEMPT    — generate solution via local LLM
       2. EVALUATE   — run tests (instrument layer — ground truth from outside the generator)
       3. RECORD     — episode in anneal-memory (typed: attempt / finding / observation / decision)
       4. CONSOLIDATE — prepare_wrap → validated_save_continuity
                        (graduation gate, immune system, citation decay)
       5. CHECK      — termination criteria (solved / max iters / stall / beyond capacity)
       6. NEXT ITERATION with compressed context, not raw history
```

**Instrument layer:** code execution. Tests pass or fail. Not the model's opinion. No LLM-as-judge.

**Frame layer:** anneal-memory's compressed-state lens on prior reasoning. Iteration N sees graduated patterns from iterations 1→N-1, not a polluted context of every failed attempt (the mem0 trap).

**Backend-agnostic.** The harness calls a `generate(prompt) → text` interface. Current backends: Ollama (Gemma 4 E4B, Qwen3.5-9B). Planned: Google AI Studio HTTP (Gemma 4 31B for cloud scale validation), llama.cpp / MLX direct (pretrained base weights for RLHF ablation).

## Why this exists

Anvil is the empirical centerpiece of an argument about harness engineering:

- **Harness intelligence > generator capability.** A patient, memory-grounded iteration loop can close capability gaps that single-shot frontier models can't. Small local model + Anvil ≥ frontier single-shot on iterative tasks. This is the thesis.
- **Memory without grounding is amplification infrastructure.** Most memory-augmented agents fail silently because they accumulate garbage. anneal-memory's immune system (citation-validated graduation, active principle demotion, anti-inbreeding) is the missing piece that makes sustained iteration work instead of collapsing into context pollution.
- **Local sovereignty.** The entire loop runs on commodity hardware (MacBook Air M4, 32GB unified memory) while the rest of a cognitive stack is also running. No cloud dependency for the core loop. Zero API cost per iteration.

## Directory layout

```
anvil/
├── README.md           — this file
├── harness/            — the pipeline code
│   ├── __init__.py
│   └── baseline.py     — single-shot (Config 0) runner
├── bugs/               — per-bug configuration files
│   ├── __init__.py
│   └── bug_01_prune_falsy.py
├── results/            — benchmark run results
│   └── bug_01_baseline.md
└── docs/               — design notes (not yet populated)
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) v0.20.2 or newer (Gemma 4 requires ≥ 0.20.2)
- At least one compatible local model — e.g.:
  ```bash
  ollama pull gemma4:e4b-it-q4_K_M   # ~9.6 GB, fits on 32GB unified memory with room to spare
  ollama pull qwen3.5:9b             # ~6.6 GB, cross-family comparison
  ```
- An [anneal-memory](https://github.com/phillipclapham/anneal-memory) checkout at `~/Documents/anneal-memory/` (Anvil's primary benchmark target). The harness refuses to run if the anneal-memory repo has uncommitted changes — it's about to mutate source files and would lose your work otherwise.

## Running the baseline

Single-shot baseline (no iteration, no memory — Config 0):

```bash
cd ~/Documents/anvil
python3 -m harness.baseline bugs.bug_01_prune_falsy gemma4:e4b-it-q4_K_M
```

The pipeline:

1. Verifies anneal-memory repo is clean (refuses to run otherwise — would risk losing work).
2. Reintroduces the bug via find/replace on the source file.
3. Runs the bug's test — expects failure (confirms reintroduction).
4. Extracts the buggy function + test source via AST.
5. Builds a prompt containing the buggy code, the failing test, and the test output.
6. Calls the Ollama API to get the model's proposed fix.
7. Extracts the fix from the model's markdown response.
8. Replaces the target function in the file via AST-based surgery (preserves class-body indentation).
9. Runs the test again.
10. Resets the anneal-memory repo via `git checkout -- .` (always, even on exception).
11. Reports pass/fail + latency + tokens + the extracted fix.

JSON output for programmatic consumption:

```bash
python3 -m harness.baseline bugs.bug_01_prune_falsy gemma4:e4b-it-q4_K_M --json
```

## First empirical finding

Running Bug 1 (`prune(older_than_days=0)` falsy short-circuit) against both models produced cross-family divergence on the easiest bug in the set:

| Model | Result | Latency |
|---|---|---|
| Gemma 4 E4B | PASS | 104.8s |
| Qwen3.5-9B | FAIL | 112.2s |

Qwen correctly identified and fixed the main bug — but silently drifted on unrelated code during whole-function regeneration, changing an SQL query operator from `<=` to `<`. That one-character drift was fatal to the test.

This is exactly the failure mode the iterative harness is designed to correct. Feed the drift back as an episode, consolidate, next iteration preserves. The mechanism the paper has been theorizing about showed up live on the first baseline run.

Full forensic: [`results/bug_01_baseline.md`](results/bug_01_baseline.md).

## Safety notes

- The pipeline mutates files in `~/Documents/anneal-memory/` during runs and resets them on exit. **It refuses to run if that repo has uncommitted changes** — this is a non-negotiable guard. Commit or stash your anneal-memory work before running Anvil.
- All state changes are contained to the target repo's working tree. Anvil does not touch `~/.anneal-memory/` (runtime data) or the anneal-memory git history.
- Every exit path — success, failure, exception — runs `git checkout -- .` to reset the target. If the reset itself fails, a warning goes to stderr and you'll need to inspect the target repo manually.

## Related work

- **[anneal-memory](https://github.com/phillipclapham/anneal-memory)** — four-layer memory (episodic store + continuity file + Hebbian associations + limbic affect) with an immune system (citation-validated graduation, active principle demotion, anti-inbreeding). Anvil's memory layer AND primary dogfood benchmark (Anvil uses anneal-memory to run on anneal-memory's own bug history).
- **Agent identity research paper** — work in progress. The theoretical frame this project is empirically testing.

## License

MIT.

---

*Part of the cognitive sovereignty stack. Built by Phillip Clapham.*
