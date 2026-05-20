# AGENTS.md — Escritura Creativa RL

## Commands

- **Run app**: `streamlit run app.py`
- **Deps already installed** in `.venv`; activate via `.venv\Scripts\activate` (Windows)

## Architecture

```
app.py → Streamlit UI
├── modo RL → rl/agent.py → env/rl_env.py → core/vocabulary.py
└── modo Bigram → core/generator.py → core/vocabulary.py
```

`core/vocabulary.py` contains a static list of ~170 unique Spanish words (no external corpus). `config.py` holds genre prefixes, max_words (30), context_size (10), temperature_default (0.7).

## Known Issues (don't fix without asking)

- **No trained PPO weights**: RL mode generates near-random text. `PPO` is instantiated but never trained (`model.learn()` never called).
- **Feedback loop incomplete**: `app.py:155-156` `reentrenar_modelo()` is a `pass`. The only feedback persistence is `logs_alineacion.json` (append via `utils/feedback.py`).
- **Missing `import torch`** in `rl/agent.py` (line 95 uses `torch.no_grad()` without import).
- **`self.agent.policy.actor`** in `rl/agent.py:97` — SB3 `ActorCriticPolicy` has `action_net`, not `.actor`. The `_get_action_probs` method will crash at runtime.

## Conventions

- All UI strings, comments, and identifiers are in Spanish
- `__init__.py` files re-export public classes via `__all__`
- Feedback: thumbs up → reward 1.0, thumbs down → -1.0
