# ChronicCompanion-AI

Research prototype for a diabetes-focused elderly companionship dialogue system with three-layer long-term memory:
- Raw interaction layer
- Mid memory layer (events + need-solution tracks)
- Top profile layer

## Prototype scope
- Not a full medical system
- No diagnosis outputs
- No high-risk treatment or medication adjustment suggestions
- Built for reproducible experiments with minimal dependencies

## Quick start
1. Copy `.env.example` to `.env` and adjust if needed.
2. Run no-memory baseline:
   - `PYTHONPATH=. python -m src.experiments.run_no_memory`
3. Run vector-memory baseline:
   - `PYTHONPATH=. python -m src.experiments.run_vector_memory_baseline`
   - Mem-PAL style batch mode:
     - `PYTHONPATH=. python -m src.experiments.run_vector_memory_baseline --dataset /data/wxf/Mem-PAL/data_synthesis_v2/data/input.json --user-start-idx 0 --user-end-idx 0 --top-k 3`
4. Run layered-memory baseline:
   - `PYTHONPATH=. python -m src.experiments.run_layered_memory`
5. Run interactive chat:
   - `PYTHONPATH=. python -m src.runner.run_chat`

## Main directories
- `src/memory`: three-layer memory schemas, stores, and update logic
- `src/retrieval`: need inference and baseline retrieval
- `src/generation`: prompt composition
- `src/safety`: risk classification and policy
- `src/experiments`: runnable baselines
- `tests`: smoke tests for prototype behavior