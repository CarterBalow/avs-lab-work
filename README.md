# AVS Lab Work

Simulation scenarios, scripts, and analysis for work with the Autonomous Vehicle Systems (AVS) Lab, built on [Basilisk](https://hanspeterschaub.info/basilisk/).

## Structure

```
.
├── scenarios/       # Basilisk scenario scripts (spacecraft, gravity, effectors, MJScene, etc.)
├── analysis/        # Post-processing, plotting, notebooks
├── notes/           # Debugging notes, design docs
└── README.md
```

## Setup

Requires a working [Basilisk](https://hanspeterschaub.info/basilisk/Install.html) installation (Python 3.9–3.11 recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a scenario

```bash
python3 scenarios/scenarioHingedRigidBody.py
```

## Notes

- Large simulation outputs (plots, logs, Vizard files) are gitignored — see `.gitignore`.
- See `notes/` for debugging logs on ongoing issues (e.g. MJScene hub-panel coupling investigation).
