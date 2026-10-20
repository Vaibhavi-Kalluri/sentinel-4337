// Added in v5: education-first simulator, including an explicitly simulated policy preview.
const originalChoose=chooseScenario;
chooseScenario=function(s){
  if(s==='blocked'){
    state.scenario='blocked';
    $$('.scenario').forEach(x=>x.classList.toggle('selected',x.dataset.scenario===s));
    const op=state.samples.unapproved||{};
    $('#scenarioDesc').textContent='Policy BLOCK preview: uses your saved unapproved-target record. The browser does not execute Solidity.';
    $('#opfields').innerHTML=Object.entries(op).map(([key,val])=>`<div class="field"><label>${esc(key.replaceAll('_',' '))} <span class="hint" title="${esc(explanations[key]||'Saved operation metadata field')}">ⓘ</span></label><span>${esc(val)}</span></div>`).join('');
    $('#resultArea').innerHTML='<div class="empty"><div class="pulse">⛔</div><h3>Blocked-target scenario ready</h3><p>Click Run to view the actual rule evaluator result alongside an explicitly simulated policy BLOCK preview.</p></div>';
  }else originalChoose(s);
};
$$('.scenario').forEach(x=>x.onclick=()=>chooseScenario(x.dataset.scenario));
const oldRun=run;
run=async function(){
  const b=$('#runBtn'); b.disabled=true; b.textContent='Checking operation…';
  try{
    const body={mode:state.scenario};
    if(state.scenario==='custom'){
      const op={...state.samples.allowed};
      $$('#opfields input').forEach(x=>{let old=op[x.dataset.key];op[x.dataset.key]=x.type==='checkbox'?x.checked:typeof old==='number'?Number(x.value):x.value});
      body.operation=op;
    }
    const response=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await response.json(); if(!response.ok)throw Error(d.error||'Unable to evaluate');
    const r=d.rule_result||{}, p=d.policy_preview||{};
    const verdict=String(r.recommendation||r.decision||'RESULT').toUpperCase();
    const reasons=r.reasons||r.triggered_rules||r.explanations||[];
    const rs=Array.isArray(reasons)?reasons:[String(reasons)];
    const score=r.risk_score??r.score??null;
    const block=p.decision==='BLOCK';
    $('#resultArea').innerHTML=`
      <div class="decision-pair"><div class="decision-card"><small>ACTUAL METADATA RULE EVALUATOR <span class="hint" title="This result is calculated by your existing Python metadata evaluator. It is advisory, not on-chain execution.">ⓘ</span></small><strong class="${verdict==='ALLOW'?'green':'amber'}">${esc(verdict)}</strong><p>${verdict==='ALLOW'?'No metadata alert from this evaluator.':'The evaluator raised a warning.'}</p></div>
      <div class="decision-card ${block?'danger':''}"><small>SIMULATED POLICY OUTCOME <span class="hint" title="Educational preview: blocks if the input marks the target unapproved. This is NOT an executed Solidity policy result.">ⓘ</span></small><strong class="${block?'red':'green'}">${esc(p.decision||'UNKNOWN')}</strong><p>${esc(p.reason||'Not evaluated')}</p></div></div>
      <div class="field"><label>Illustrative metadata risk score <span class="hint" title="From hand-written rules, not the trained 17-feature Random Forest.">ⓘ</span></label><span>${score===null?'Not returned':esc(score)}</span></div>
      <div class="field"><label>Actual blockchain transaction</label><span>Not submitted</span></div>
      <div class="timeline"><div><b>01 · Input selected</b><small>${esc(d.source)}</small></div><div><b>02 · Existing metadata rules executed</b><small>${rs.length?rs.map(esc).join(' · '):'No individual reasons returned.'}</small></div><div><b>03 · Demonstration policy checked</b><small>${esc(p.rule||'No preview available')}</small></div><div><b>04 · What would happen?</b><small>${block?'In this educational policy preview, the unapproved destination would be rejected.':'In this educational policy preview, the destination is not rejected by the approval rule.'} No on-chain decision was executed.</small></div></div>
      <details><summary>View raw Python evaluator output</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
  }catch(e){$('#resultArea').innerHTML=`<div class="callout">${esc(e.message)}</div>`}
  finally{b.disabled=false;b.textContent='▶ Run security simulation'}
};
$('#runBtn').onclick=run;
// Let the custom scenario emphasize the exact field that changes the preview outcome.
const selectOriginal=chooseScenario;
chooseScenario=function(s){selectOriginal(s);if(s==='custom'){
  $('#scenarioDesc').innerHTML='Edit the values, especially <b>is approved target</b>. Turn it off to see a <b>simulated BLOCK</b>; turn it on to see a simulated ALLOW. The actual metadata evaluator still returns its own recommendation.';
}};
$$('.scenario').forEach(x=>x.onclick=()=>chooseScenario(x.dataset.scenario));
