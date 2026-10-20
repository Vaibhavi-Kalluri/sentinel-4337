# Sentinel-4337 — Prototype Integration (Iteration 1)

This patch adds a **local** unified decision layer to the existing Flask v7 UI.

## What is new
- `monitoring/prototype_decision.py`: transparent LOW/MEDIUM/HIGH policy logic with ALLOW, REVIEW, BLOCK, GATE.
- `server.py`: `POST /api/prototype/check`, calls the existing trained Random Forest evaluator (17 features) and returns separate ML, thresholds, and simulated policy outputs.
- `static/prototype.js`: new panel in the Smart Wallet Simulator; high-tier operations can complete an explicitly **simulated** confirmation.
- `templates/index.html` and `static/style.css`: load and style the panel.
- `tests/test_prototype_decision.py`: six unit tests.

## Safety and research boundaries
- This is **not** a cryptographic ZK gate or real blockchain enforcement.
- Browser inputs are not trusted transaction traces; a scenario's recipient/amount does not directly change the ML score.
- Existing `monitoring/ml_evaluator.py` still uses `0.5` for its advisory recommendation.
- The new decision layer uses `threshold_selected_on_validation` from `reports/real_background_benchmark/report.json` (approximately `0.44663`) for MEDIUM review, and `0.8` as an **illustrative** HIGH cutoff (not validated or calibrated).
- Saved benchmark F1 applies to controlled feature perturbations, not verified malicious transactions.

## Apply to Kali repository
Extract this ZIP **over** `~/sentinel-4337` after making a backup. ZIP entries are relative to the repository root. It contains only modified/new files, not the large model files.

```bash
cd ~
cp -a sentinel-4337 sentinel-4337-pre-prototype-backup
unzip -o /path/to/sentinel4337_prototype_patch.zip -d ~/sentinel-4337
cd ~/sentinel-4337
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_prototype_decision.py' -v
python server.py
```

Open http://127.0.0.1:5050, navigate to Smart Wallet Simulator, set the recipient and trace signals, then click **Evaluate unified prototype**.

## Test results in the patch environment
- 6 / 6 unit tests passed.
- `python -m py_compile server.py monitoring/prototype_decision.py` passed.
- Full model-inference and Solidity integration tests were **not run** in this environment; run those on Kali before committing.

## Safe Git workflow after verification
```bash
git status -sb
git diff --check
git diff -- server.py templates/index.html static/style.css
git add server.py monitoring/prototype_decision.py static/prototype.js static/style.css templates/index.html tests/test_prototype_decision.py PROTOTYPE-UPDATE.md
# Inspect the staged changes before committing:
git diff --cached --stat
git commit -m "Add unified local security decision prototype"
git push origin HEAD
```

Do not push until the UI, Python tests, and existing Foundry tests pass on your machine.
