from pathlib import Path
from flask import Flask, jsonify, request, render_template
import json, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parent
app=Flask(__name__,template_folder='templates',static_folder='static')
BASE=ROOT/'reports/real_background_benchmark/report.json'
SAMPLES={'allowed':'sample_allowed.json','unapproved':'sample_unapproved.json','approval':'sample_approval.json'}

def report():
    return json.loads(BASE.read_text()) if BASE.exists() else {}

@app.get('/')
def home(): return render_template('index.html')

@app.get('/api/summary')
def summary():
    r=report()
    return jsonify({'report':r,'samples':{k:json.loads((ROOT/'monitoring'/v).read_text()) for k,v in SAMPLES.items() if (ROOT/'monitoring'/v).exists()}})

@app.post('/api/simulate')
def simulate():
    data=request.get_json(silent=True) or {}
    mode=data.get('mode','allowed')
    if mode not in [*SAMPLES,'custom','blocked']: return jsonify(error='Invalid scenario'),400
    if mode=='custom':
        op=data.get('operation')
        if not isinstance(op,dict):return jsonify(error='Provide an operation object'),400
    else:
        p=ROOT/'monitoring'/SAMPLES['unapproved' if mode=='blocked' else mode]
        if not p.exists():return jsonify(error='Saved sample not found'),404
        op=json.loads(p.read_text())
    # Existing rule-based evaluator, separate from the ML trace-feature evaluator.
    script=ROOT/'monitoring/evaluate_operation.py'
    if not script.exists():return jsonify(error='Rule evaluator not found'),404
    tmp=None
    try:
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
            json.dump(op,f);tmp=Path(f.name)
        run=subprocess.run([sys.executable,str(script),str(tmp)],cwd=ROOT,capture_output=True,text=True,timeout=15)
        if run.returncode:return jsonify(error=(run.stderr or run.stdout)[-900:]),400
        result=json.loads(run.stdout)
        # Clearly labeled educational policy preview, NOT a call to the Solidity contract.
        approved=op.get('is_approved_target')
        if isinstance(approved,bool):
            preview={'decision':'ALLOW' if approved else 'BLOCK',
                     'reason':'Target is marked approved in supplied metadata.' if approved else 'Target is marked unapproved in supplied metadata.',
                     'rule':'Demonstration rule: block when is_approved_target is false',
                     'type':'SIMULATED_POLICY_PREVIEW', 'on_chain':False}
        else:
            preview={'decision':'UNKNOWN','reason':'No is_approved_target boolean in this record.',
                     'rule':'No policy preview possible','type':'SIMULATED_POLICY_PREVIEW','on_chain':False}
        return jsonify(input=op,rule_result=result,policy_preview=preview,
            source=('Saved '+mode+' example' if mode!='custom' else 'Custom user input'),
            enforcement='Not executed on-chain',ml='Not evaluated: this sample uses metadata, not trace-derived ML features')
    except (ValueError,json.JSONDecodeError) as e:return jsonify(error=str(e)),400
    except subprocess.TimeoutExpired:return jsonify(error='Evaluator timed out'),504
    finally:
        if tmp:tmp.unlink(missing_ok=True)

@app.post('/api/ml')
def ml():
    op=(request.get_json(silent=True) or {}).get('features',{})
    try:
        from monitoring.ml_evaluator import evaluate
        return jsonify(result=evaluate(op),enforcement='Not executed on-chain')
    except Exception as e:return jsonify(error=str(e)),400

@app.post('/api/prototype/check')
def prototype_check():
    """Unified local decision: actual ML inference + simulated policy/gate."""
    from monitoring.prototype_decision import decide
    body=request.get_json(silent=True)
    if not isinstance(body,dict): return jsonify(error='JSON object required'),400
    if type(body.get('approved_target')) is not bool:
        return jsonify(error='approved_target must be a boolean'),400
    features=body.get('features')
    if not isinstance(features,dict):return jsonify(error='features must be an object'),400
    try:
        from monitoring.ml_evaluator import evaluate
        ml_result=evaluate(features)
        benchmark=report()
        threshold=benchmark.get('threshold_selected_on_validation',0.4466300357476273)
        decision=decide(approved_target=body['approved_target'],ml_score=ml_result['risk_score'],review_threshold=threshold)
        return jsonify(ml=ml_result,decision=decision,thresholds={'review':threshold,'high_demo':0.8,'existing_evaluator':0.5},
            disclosure='Local prototype only. No ZK proof, wallet signature, chain submission or Solidity enforcement.',
            data_origin='User-supplied feature counts; not a trusted EVM trace.')
    except (ValueError,TypeError,FileNotFoundError) as e:
        return jsonify(error=str(e)),400

if __name__=='__main__':app.run(host='127.0.0.1',port=5050,debug=False)
