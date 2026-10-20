from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Sentinel 4337 • Security Studio',page_icon='🛡️',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 85% 0%,#18304a 0%,#080e19 43%);color:#f0f6ff}
html,body,[class*=css]{font-family:'DM Sans',sans-serif} [data-testid=stSidebar]{background:#0c1524;border-right:1px solid #22344a}
h1,h2,h3{letter-spacing:-.04em} h1{font-weight:800} p{color:#bdcbda}
[data-testid=stMetric]{background:#101f30;border:1px solid #284057;border-radius:17px;padding:17px}
[data-testid=stMetricValue]{color:#f0f9ff} [data-testid=stMetricLabel]{color:#a7bbcd}
.stButton>button[kind=primary]{background:#39d6c1;color:#071724;border:0;font-weight:800;border-radius:12px}
.stButton>button{border-radius:12px} [data-testid=stDataFrame]{border:1px solid #284057;border-radius:12px}
.hero{background:linear-gradient(110deg,#18344a,#101a2a);border:1px solid #2b5264;border-radius:22px;padding:28px;margin-bottom:22px}
.hero h1{margin:5px 0 9px;font-size:2.3rem} .hero p{margin:0;max-width:800px}
.kicker{font-size:12px;color:#58e7d5;font-weight:800;letter-spacing:2px}
.explain{padding:15px 18px;border-radius:12px;background:#122638;border:1px solid #2a5064;color:#c7dbe8;margin:12px 0}
.warn{padding:14px 17px;border-radius:12px;background:#292315;border:1px solid #67512d;color:#f7d69b;margin:12px 0}
.result{padding:20px;border:1px solid #32516b;background:#112235;border-radius:18px}
small{color:#aabccc}
</style>''',unsafe_allow_html=True)

def load(path):
    try:return json.loads((ROOT/path).read_text(encoding='utf-8'))
    except Exception:return None

def pct(x):return f'{100*float(x):.2f}%'
def hero(tag,title,subtitle):st.markdown(f'<div class="hero"><div class="kicker">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)
def note(s,warning=False):st.markdown(f'<div class="{"warn" if warning else "explain"}">{s}</div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 🛡️ SENTINEL 4337')
    st.caption('SECURITY INTELLIGENCE STUDIO')
    st.divider()
    page=st.radio('Explore', ['📊 Research comparison','🛡️ Account simulator','📈 Performance insights','🧩 Model architecture','⚙️ How it works'],label_visibility='collapsed')
    st.divider()
    st.caption('Evidence-first demonstration • saved results + existing model')

r=load('reports/real_background_benchmark/report.json') or {}
if page.startswith('📊'):
    hero('01 / RESEARCH','Research vs Sentinel-4337','Compare published Ethereum phishing detection research with the benchmark from your own ERC-4337 security project.')
    if not r:st.error('Benchmark report not found. Place app.py in the Kali project root alongside reports/.');st.stop()
    a,b,c,d=st.columns(4)
    a.metric('Our F1-score',pct(r['test_f1']),help='F1 balances precision (how many alerts were correct) and recall (how many perturbed examples were detected).')
    b.metric('Our precision',pct(r['test_precision']),help='Among examples flagged by the model, the percentage that had controlled perturbation labels.')
    c.metric('Our recall',pct(r['test_recall']),help='Among all perturbed test examples, the percentage correctly flagged by the model.')
    d.metric('Our accuracy',pct(r['classification_report']['accuracy']),help='Percentage of test examples correctly classified; does not mean real-world attack detection accuracy.')
    st.markdown('### Published results and our benchmark')
    df=pd.DataFrame([
        {'Study':'DiT-SGCR (2025)','Dataset':'MulDiGraph','F1 (%)':91.56,'Type':'Phishing account detection'},
        {'Study':'DiT-SGCR (2025)','Dataset':'B4E','F1 (%)':83.01,'Type':'Phishing account detection'},
        {'Study':'DiT-SGCR (2025)','Dataset':'Transactions Network','F1 (%)':90.43,'Type':'Phishing account detection'},
        {'Study':'FAAN-GBM (2024)','Dataset':'EthereumData1','F1 (%)':99.54,'Type':'Phishing account detection'},
        {'Study':'FAAN-GBM (2024)','Dataset':'EthereumData2','F1 (%)':99.24,'Type':'Phishing account detection'},
        {'Study':'Sentinel-4337','Dataset':'Controlled perturbation benchmark','F1 (%)':round(100*r['test_f1'],2),'Type':'Feature perturbation classification'},
    ])
    st.bar_chart(df.assign(Comparison=df['Study']+' · '+df['Dataset']).set_index('Comparison')[['F1 (%)']],horizontal=True,x_label='Reported F1-score (%)',color='#42d9c7')
    with st.expander('View exact values and paper sources'):
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.markdown('**Sources:** DiT-SGCR (2025), Table II; FAAN-GBM (2024), Tables 3–4; Sentinel-4337, `reports/real_background_benchmark/report.json`.')
    note('<b>How to interpret:</b> These are published and locally measured scores on <b>different datasets and tasks</b>. They are useful for research context, not proof that one system outperforms another.',True)
    st.markdown('### What makes Sentinel-4337 different?')
    c1,c2,c3=st.columns(3)
    with c1:st.info('**Research papers**\n\nStudy suspicious Ethereum accounts using transaction relationships and learned account representations.')
    with c2:st.info('**Sentinel ML**\n\nScores trace-derived ERC-4337 operation features using an existing Random Forest model.')
    with c3:st.info('**Sentinel policy**\n\nDeterministic Solidity rules are a separate enforcement mechanism; ML scoring is advisory.')

elif page.startswith('🛡️'):
    hero('02 / INTERACTIVE DEMO','Account operation simulator','Select a real saved sample or enter trace-feature counts, then run your existing Random Forest evaluator.')
    samples={}
    for p in sorted((ROOT/'monitoring').glob('sample*.json')):
        obj=load(str(p.relative_to(ROOT)))
        if isinstance(obj,dict):samples[p.stem.replace('sample_','').replace('_',' ').title()]=obj
    options=['Custom operation']+list(samples)
    chosen=st.selectbox('Operation input',options)
    selected=samples.get(chosen,{})
    keys=['trace_count','unique_destinations','approve_trace_count','transfer_from_trace_count','value_transfer_trace_count','errored_trace_count']
    values={}
    st.markdown('#### Step 1 · Choose an example')
    st.info('**What is happening?** An ERC-4337 account operation can trigger internal blockchain actions. Your model does not read a wallet address here; it reads counts describing those actions. Choose a saved example above, or use Custom operation to explore. Then press **Analyze operation** to run the saved ML model.')
    with st.expander('What do the input numbers mean? Start here',expanded=False):
        st.markdown('**Trace events:** internal execution actions. **Destinations:** different addresses touched. **Approval events:** token spending permissions. **Transfer-from events:** token transfers initiated using an allowance. **Value-transfer events:** actions that move native currency. **Errored events:** internal actions that failed. **Paymaster:** an entity that may sponsor transaction gas. These counts describe behavior; none alone proves an attack.')
    st.markdown('#### Step 2 · Review operation activity')
    st.caption('Hover over the ⓘ icons next to inputs to understand them. All values describe trace features, not real-time wallet activity.')
    left,right=st.columns(2)
    names=['Total trace events','Unique destinations','Approval events','Transfer-from events','Value-transfer events','Errored trace events']
    helps={
      'trace_count':'Total internal trace events associated with the operation. A larger number means more observed internal activity, not automatically higher danger.',
      'unique_destinations':'Number of distinct destinations touched in the trace. This is not the number of different wallets owned by an account.',
      'approve_trace_count':'Count of token approval-related trace events. An approval gives permission to spend tokens, but is not automatically malicious.',
      'transfer_from_trace_count':'Count of transferFrom-related events, which can move tokens under an allowance.',
      'value_transfer_trace_count':'Count of native value transfer events in the trace.',
      'errored_trace_count':'Count of trace events marked as errors or failures.'}
    for i,(key,label) in enumerate(zip(keys,names)):
        initial=selected.get(key,0)
        if not isinstance(initial,int) or isinstance(initial,bool):initial=0
        with (left if i%2==0 else right):values[key]=st.number_input(label,min_value=0,value=initial,step=1,key=f'{chosen}_{key}',help=helps[key])
    pay=st.checkbox('Uses a paymaster',value=bool(selected.get('has_paymaster',False)),key=f'pay_{chosen}',help='A paymaster may sponsor the gas fee for an ERC-4337 operation. This alone is not a security verdict.')
    operation={**values,'has_approval':int(values['approve_trace_count']>0),'has_transfer_from':int(values['transfer_from_trace_count']>0),'has_value_transfer':int(values['value_transfer_trace_count']>0),'has_trace_errors':int(values['errored_trace_count']>0),'has_paymaster':int(pay)}
    st.markdown('#### Step 3 · Run the saved model')
    if st.button('🛡️ Analyze operation',type='primary',use_container_width=True):
        try:
            sys.path.insert(0,str(ROOT))
            from monitoring.ml_evaluator import evaluate
            st.session_state['latest']=(operation,evaluate(operation))
        except Exception as exc:st.error(f'Model evaluation failed: {exc}')
    if 'latest' in st.session_state:
        inp,result=st.session_state['latest'];score=float(result.get('risk_score',0));flag=result.get('recommendation','UNKNOWN')
        st.markdown('### Analysis result')
        x,y,z=st.columns(3)
        x.metric('Model risk score',f'{score:.3f}',help='A Random Forest output between 0 and 1. This is NOT a calibrated probability of maliciousness.')
        y.metric('ML recommendation',flag.replace('_',' '),help='REVIEW means the ML score met its advisory threshold; NO ML ALERT means it did not. Neither is an on-chain block/allow decision.')
        z.metric('On-chain decision','Not executed',help='The UI does not call your Solidity policy or submit a UserOperation. Only a separate policy execution can confirm ALLOW or BLOCK.')
        st.progress(min(1,max(0,score)))
        st.caption('Score closer to 1 = more similar to the benchmark’s perturbed patterns. The score is advisory, not an attack probability.')
        if flag=='REVIEW':note('⚠️ <b>Review recommended.</b> The ML model flagged this feature pattern. This is <b>not</b> proof of an attack or an on-chain block.',True)
        else:note('✓ <b>No ML alert.</b> The model did not flag this pattern at its inference threshold. This is <b>not</b> an on-chain approval.')
        st.markdown('#### What was analyzed?')
        st.dataframe(pd.DataFrame([{'Activity':n,'Count':inp[k]} for n,k in zip(names,keys)]),hide_index=True,use_container_width=True)
        with st.expander('Technical details / raw model response'):st.json(result)
    note('The model returns REVIEW or NO_ML_ALERT. It does not itself allow, block, submit, or simulate a UserOperation on-chain. A confirmed ALLOW/BLOCK requires a separate real policy execution.',True)

elif page.startswith('📈'):
    hero('03 / EXPERIMENTS','Performance insights','Understand your benchmark through clear counts, charts and plain-English explanations.')
    if not r:st.error('Saved benchmark report not found.');st.stop()
    n=r['test_counts'];total=sum(n.values())
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Test examples',f'{total:,}')
    c2.metric('Correct predictions',f'{n["tp"]+n["tn"]:,}')
    c3.metric('False alarms',f'{n["fp"]:,}')
    c4.metric('Missed perturbations',f'{n["fn"]:,}')
    st.markdown('### Correct decisions vs mistakes')
    counts=pd.DataFrame([{'Outcome':'Correct: unmodified background','Count':n['tn']},{'Outcome':'Correct: flagged perturbation','Count':n['tp']},{'Outcome':'Mistake: false alarm','Count':n['fp']},{'Outcome':'Mistake: missed perturbation','Count':n['fn']}])
    st.bar_chart(counts.set_index('Outcome')[['Count']],horizontal=True,color='#44c8e5')
    note('Out of <b>16,800 test examples</b>, the model correctly classified <b>15,830</b> and misclassified <b>970</b>. The positive class is controlled synthetic feature perturbations, not confirmed fraud.')
    st.markdown('### Quality and speed')
    a,b,c=st.columns(3)
    a.metric('False positive rate',pct(r['test_false_positive_rate']))
    b.metric('ROC-AUC',pct(r['test_roc_auc']))
    c.metric('Inference per example',f'{r["test_inference_ms_per_row"]:.4f} ms')
    st.caption('Inference timing is from the saved benchmark environment, not a guaranteed deployment latency.')
    with st.expander('Benchmark details'):
        st.write(f"Training: {r['train_rows']:,} • Validation: {r['validation_rows']:,} • Test: {r['test_rows']:,} • Features: {r['feature_count']}")
        st.write('Validation-selected threshold:',r['threshold_selected_on_validation'])
        st.json(r.get('limitations',[]))

elif page.startswith('🧩'):
    hero('04 / ARCHITECTURE','How we built the model','A guided tour of your actual Sentinel-4337 machine-learning pipeline, from saved data to an advisory risk score.')
    st.markdown('### The ML pipeline — from data to decision')
    steps=[
      ('01','Real blockchain background','Existing operation/trace-derived records form the background for the benchmark.','This does not mean every background record is verified benign.'),
      ('02','Controlled feature perturbations','The benchmark modifies selected aggregate features to create a second class for evaluation.','Perturbations are not confirmed real attacks; transaction traces were not reconstructed.'),
      ('03','Feature engineering · 17 inputs','Six count features + five binary indicators + six log-transformed counts.','Example: approval event count becomes an approval-present flag; log transforms reduce the influence of very large counts.'),
      ('04','Train Random Forest','A Random Forest learns patterns separating unmodified background records from controlled perturbations.','The saved model is used for inference; opening this UI does not retrain it.'),
      ('05','Validate and test','50,400 train rows, 16,800 validation rows and 16,800 test rows; the evaluation threshold was selected on validation.','Holdout performance describes this benchmark, not real-world maliciousness detection.'),
      ('06','Inference · advisory output','The UI sends your input features to monitoring/ml_evaluator.py and receives a score plus REVIEW / NO_ML_ALERT.','The score is not a calibrated attack probability and does not block a transaction.'),
    ]
    for number,name,description,detail in steps:
        with st.container(border=True):
            a,b=st.columns([1,9]);a.markdown(f'### {number}');b.markdown(f'**{name}**');b.write(description);b.caption('ⓘ '+detail)
    st.markdown('### What happens after ML?')
    a,b=st.columns(2)
    with a:
        st.success('**Implemented separately: Solidity policy**\n\nDeterministic policy checks can allow or revert an operation when executed on-chain. This dashboard does not invoke those checks.')
    with b:
        st.warning('**Proposed only: Zero-knowledge gate**\n\nA future privacy-preserving verification concept. Not implemented as a working proof system in this demo.')
    st.markdown('### Your benchmark, in numbers')
    if r:
        a,b,c,d=st.columns(4)
        a.metric('Training rows',f"{r.get('train_rows',0):,}",help='Records used to fit the classifier')
        b.metric('Validation rows',f"{r.get('validation_rows',0):,}",help='Records used to choose a threshold')
        c.metric('Test rows',f"{r.get('test_rows',0):,}",help='Held-out examples used to report test metrics')
        d.metric('Model inputs',str(r.get('feature_count','—')),help='6 raw counts + 5 binary flags + 6 log1p transformed counts')
        with st.expander('See all 17 exact feature names'):st.write(r.get('features',[]))
    st.caption('Relevant code: monitoring/ml_evaluator.py · Saved model: models/sentinel_real_background_injected_rf.joblib · Benchmark: reports/real_background_benchmark/report.json')

else:
    hero('05 / SYSTEM','How Sentinel-4337 works','A simple explanation of what is implemented, what is advisory, and what remains a research concept.')
    st.markdown('### From operation to security decision')
    st.markdown('**① ERC-4337 UserOperation** → **② Trace-derived features** → **③ ML advisory risk score**')
    st.markdown('**④ Deterministic SentinelPolicy rules** → **⑤ On-chain allow/revert outcome when the policy is actually executed**')
    note('The ML evaluator and Solidity policy are separate components. A model recommendation must not be displayed as an executed policy decision.')
    st.markdown('### Components')
    st.markdown('**Random Forest evaluator:** reads 17 engineered features and returns an advisory score.\n\n**Solidity policy:** implements deterministic checks; on-chain outcomes require an actual policy execution.\n\n**ZK gate:** proposed architecture / future work, not an implemented proof-verification gate.')
    st.markdown('### Research limitations')
    st.write('The benchmark tests controlled synthetic perturbations of features derived from real-background records. It does not establish real-world malicious-operation detection accuracy.')
