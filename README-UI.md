# Sentinel-4337 UI bundle

## Install / run (from the repository root)
Copy `app.py` into `~/sentinel-4337/app.py` and `requirements-ui.txt` into `~/sentinel-4337/requirements-ui.txt`, then:

```bash
cd ~/sentinel-4337
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Design and data integrity
- Overview reads existing CSV/JSON artifacts from `reports/` and model file counts from `models/`.
- Analyze an Operation calls `monitoring.ml_evaluator.evaluate()` and uses `models/sentinel_real_background_injected_rf.joblib`.
- Architecture page explicitly marks the ZK gate as a concept unless separately implemented and verified.
- The model score is not a calibrated attack probability; benchmark data uses controlled perturbation patterns, not independently verified malicious attacks.
- No transaction is submitted and no on-chain enforcement is invoked by this UI.
