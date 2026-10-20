// v7 scenario storytelling. Uses existing real Python evaluation endpoints; never fabricates ML outputs.
(()=>{
const scenarios=[
 {id:'normal',icon:'✓',name:'Normal payment',short:'Routine approved transfer',action:'transfer',recipient:'approved',amount:.2,counts:[3,1,0,0,1,0],explain:'A standard transfer to an approved recipient. The destination rule should pass; the ML score is still calculated from trace-feature inputs.'},
 {id:'substitution',icon:'↗',name:'Changed recipient',short:'Destination substitution',action:'transfer',recipient:'unapproved',amount:1.2,counts:[4,2,0,0,1,0],explain:'The requested destination is not approved. The deterministic policy preview should show BLOCK, regardless of the ML advisory.'},
 {id:'approval',icon:'∞',name:'Token permission',short:'Potential unlimited approval',action:'approval',recipient:'unapproved',amount:0,counts:[9,3,5,0,0,0],explain:'A token approval can grant another contract spending permission. This scenario illustrates why permissions require review; the current simulator does not decode actual token allowances.'},
 {id:'drain',icon:'⇢',name:'Repeated transfers',short:'Wallet-drain pattern',action:'batch',recipient:'unapproved',amount:3.8,counts:[25,6,0,8,10,2],explain:'Many value transfers and transferFrom events can be concerning. These counts are illustrative model inputs, not an actual reconstructed attack trace.'},
 {id:'module',icon:'⚙',name:'Wallet module',short:'Malicious add-on concept',action:'batch',recipient:'unapproved',amount:0,counts:[13,5,1,2,0,2],explain:'A malicious module could alter wallet behavior. The current 17-feature model does not directly identify module installation; this is an architectural threat example.'}
];
const grid=document.getElementById('threatGrid');if(!grid)return;
const $=id=>document.getElementById(id);
let selected='normal';
function updateNarrative(){let action=$('walletAction').selectedOptions[0].text,recipient=$('walletRecipient').value==='approved'?'approved':'unapproved',amount=$('walletAmount').value;$('operationNarrative').textContent=`You are preparing: ${action}, amount ${amount} ETH, to an ${recipient} destination. In a real ERC-4337 flow the signed UserOperation would be submitted through a bundler and EntryPoint. Here we only evaluate local sample metadata and supplied trace-feature counts.`;}
function pick(s){selected=s.id;grid.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.scenario===s.id));$('walletAction').value=s.action;$('walletAction').dispatchEvent(new Event('change'));$('walletRecipient').value=s.recipient;$('walletAmount').value=s.amount;s.counts.forEach((v,i)=>{const ids=['trace_count','unique_destinations','approve_trace_count','transfer_from_trace_count','value_transfer_trace_count','errored_trace_count'];const el=$('sig_'+ids[i]);if(el){el.value=v;el.dispatchEvent(new Event('input'));}});$('threatExplanation').textContent=s.explain;updateNarrative();}
grid.innerHTML=scenarios.map(s=>`<button type="button" class="threat-option" data-scenario="${s.id}" title="${s.explain}"><span>${s.icon}</span><b>${s.name}</b><small>${s.short}</small></button>`).join('');grid.querySelectorAll('button').forEach((b,i)=>b.addEventListener('click',()=>pick(scenarios[i])));
['walletAction','walletRecipient','walletAmount'].forEach(id=>$(id)?.addEventListener('change',updateNarrative));pick(scenarios[0]);
})();
