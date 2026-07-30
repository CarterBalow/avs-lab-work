# AVS Lab Work

Simulation scenarios, scripts, and analysis for work with the Autonomous Vehicle Systems (AVS) Lab, built on [Basilisk](https://hanspeterschaub.info/basilisk/). Also includes experimental work for electroadhesion testing in atmosphere and vaccuum.

## Structure

```
.
├── scenarios/       # Basilisk scenario scripts (spacecraft, gravity, effectors, MJScene, etc.)
├── experimental/    # Experiment scripts, plots, etc.
└── README.md
```

## Setup
FOR BASILISK CODE:
Requires a working [Basilisk](https://hanspeterschaub.info/basilisk/Install.html) installation (Python 3.12 recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FOR EXPERIMENT CODE:
Currently requires working installations for:
- Bota Systems' [bota-driver](https://code.botasys.com/en/gen_a/layer1/driver/driver.html)
- Xeryon's [Python Software](https://xeryon.com/software/xeryon-python-library/)

## Running a scenario

```bash
python3 scenarios/scenarioHingedRigidBodyMuJoCo.py
```

## Notes

- Large simulation outputs (plots, logs, Vizard files) are gitignored — see `.gitignore`.
- Some files are translations of native Basilisk files to incorporate MuJoCo dynamics
