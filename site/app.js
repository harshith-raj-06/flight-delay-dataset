'use strict';
const pages={'/':'Overview','/protocol':'Data and protocol','/models':'Model comparison','/features':'Feature groups','/interpretation':'Interpretation','/errors':'Errors and subgroups','/explorer':'Risk explorer','/test':'Final test','/method':'Method and limitations'};
const path=location.pathname, content=document.querySelector('#content');
const names={xgboost:'XGBoost',lightgbm:'LightGBM',histgb:'Histogram GB',pca_logistic_regression:'PCA + logistic',logistic_regression:'Logistic regression',gradient_boosting:'Gradient boosting',random_forest:'Random forest',decision_tree:'Decision tree',majority:'Majority prior',adaboost:'AdaBoost',catboost:'CatBoost'};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s), pct=x=>(100*x).toFixed(1)+'%', dec=x=>Number(x).toFixed(4);
const section=(h,body)=>'<section><h2>'+esc(h)+'</h2>'+body+'</section>', note=t=>'<p class="notice">'+esc(t)+'</p>';
const options=xs=>xs.map(x=>'<option value="'+esc(x)+'">'+esc(names[x]||x)+'</option>').join('');
const get=async u=>{const r=await fetch(u);if(!r.ok)throw Error('Could not load '+u);return r.json()};
const data=k=>get('/data/'+k+'.json');
let tables=[];
function table(rows,label){
 if(!rows.length)return note('No matching records.');
 const id=tables.push({rows:[...rows],label})-1;
 return '<div><button class="download" data-export="'+id+'">Download CSV</button><div class="scroll"><table aria-label="'+esc(label)+'" data-id="'+id+'"></table></div></div>';
}
function drawTables(){document.querySelectorAll('table[data-id]').forEach(el=>{
 const id=Number(el.dataset.id),{rows}=tables[id],keys=Object.keys(rows[0]);
 el.innerHTML='<thead><tr>'+keys.map((k,i)=>'<th><button data-sort="'+i+'">'+esc(k.replaceAll('_',' '))+'</button></th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(names[r[k]]||r[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';
});}
document.addEventListener('click',e=>{
 const sort=e.target.dataset.sort, exp=e.target.dataset.export;
 if(sort!==undefined){const t=tables[e.target.closest('table').dataset.id],k=Object.keys(t.rows[0])[+sort];t.dir=t.key===k?-t.dir:1;t.key=k;t.rows.sort((a,b)=>t.dir*String(a[k]).localeCompare(String(b[k]),undefined,{numeric:true}));drawTables();}
 if(exp!==undefined){const {rows,label}=tables[exp],keys=Object.keys(rows[0]),csv=[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download=label+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
});
function bars(rows,key,value,sd,cap){
 return '<div class="bars" role="img" aria-label="Horizontal comparison chart, data in the following table">'+rows.map(r=>{
 const v=Number(r[value]),s=sd?Number(r[sd]):0;
 return '<div class="bar" title="'+esc(r[key])+': '+v+(sd?' ± '+s:'')+'"><span>'+esc(names[r[key]]||r[key])+'</span><div class="track"><div class="fill" style="width:'+Math.min(100,v/cap*100)+'%"></div>'+(sd?'<i class="range" style="left:'+Math.max(0,(v-s)/cap*100)+'%;width:'+2*s/cap*100+'%"></i>':'')+'</div><span>'+dec(v)+'</span></div>';
 }).join('')+'</div>';
}
function chart(title,caption,svg,rows){
 return '<figure class="plot"><h3>'+esc(title)+'</h3>'+svg+'<figcaption>'+esc(caption)+'</figcaption><details><summary>Chart data and CSV</summary>'+table(rows,title)+'</details></figure>';
}
function lineChart(rows,x,y,opts){
 const {title,caption,xLabel,yLabel,minX,maxX,minY,maxY,percent=false,diagonal=false}=opts;
 const sorted=[...rows].sort((a,b)=>+a[x]-b[x]);
 const X=v=>68+480*(v-minX)/(maxX-minX),Y=v=>192-155*(v-minY)/(maxY-minY);
 const label=v=>percent?Math.round(v*100)+'%':Number(v.toPrecision(3)).toString();
 const ticks=(a,b,n)=>Array.from({length:n+1},(_,i)=>a+i*(b-a)/n);
 let svg='<svg viewBox="0 0 580 242" role="img" aria-label="'+esc(title)+'"><text x="10" y="16">'+esc(yLabel)+'</text>';
 for(const v of ticks(minY,maxY,4))svg+='<path class="grid" d="M68 '+Y(v)+'H548"/><text x="60" y="'+(Y(v)+4)+'" text-anchor="end">'+label(v)+'</text>';
 for(const v of (opts.xTicks||ticks(minX,maxX,4)))svg+='<text x="'+X(v)+'" y="213" text-anchor="middle">'+((diagonal||opts.baseline!=null)?label(v):Number(v.toPrecision(3)))+'</text>';
 if(opts.baseline!=null)svg+='<path class="reference" d="M68 '+Y(opts.baseline)+'H548"/>';
 if(diagonal)svg+='<path class="reference" d="M68 192L548 37"/>';
 svg+='<polyline class="series" points="'+sorted.map(r=>X(+r[x])+','+Y(+r[y])).join(' ')+'"/>';
 for(const r of sorted)svg+='<circle cx="'+X(+r[x])+'" cy="'+Y(+r[y])+'" r="'+(sorted.length>30?2:4)+'"><title>'+esc(x+' '+r[x]+', '+y+' '+r[y]+(r.flights?', flights '+r.flights:''))+'</title></circle>';
 return chart(title,caption,svg+'<text x="310" y="238" text-anchor="middle">'+esc(xLabel)+'</text></svg>',sorted);
}
function heatmap(rows){
 const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
 const scale=Math.max(...rows.map(r=>r.late_rate));
 let svg='<svg viewBox="0 0 610 212" role="img" aria-label="Late rate by weekday and local departure hour">';
 for(const r of rows)svg+='<rect x="'+(44+r.dep_hour*23)+'" y="'+(8+r.dow*23)+'" width="21" height="21" fill="rgba(17,71,83,'+r.late_rate/scale+')"><title>'+days[r.dow]+' '+r.dep_hour+'h: '+pct(r.late_rate)+' late, '+r.flights+' flights</title></rect>';
 days.forEach((v,i)=>svg+='<text x="3" y="'+(23+i*23)+'">'+v+'</text>');
 for(let h=0;h<24;h+=3)svg+='<text x="'+(45+h*23)+'" y="188">'+h+'</text>';
 return chart('How delay rates vary through the week','Darker teal means a higher fraction delayed. Each cell pools development flights at the same origin-local hour and weekday; hover for counts.',svg+'<text x="300" y="209" text-anchor="middle">Scheduled departure hour (local)</text></svg>',rows);
}
function show(html){content.innerHTML=html;drawTables()}
$('#nav').innerHTML=Object.entries(pages).map(([url,title])=>'<a href="'+url+'" '+(url===path?'aria-current="page"':'')+'>'+title+'</a>').join('');
$('#menu').onclick=()=>$('#menu').setAttribute('aria-expanded',$('#nav').classList.toggle('open'));
$('#title').textContent=pages[path];document.title=pages[path]+' · Departure studies';
$('#period').textContent=path==='/test'?'Final test · 1–31 December 2025':path==='/explorer'?'Development refit · Interactive prediction':['/models','/errors','/'].includes(path)?'Validation · Folds 3–4 · 13 Oct – 28 Nov 2025':'2025 study · Two-hour checkpoint';
function studyDependenceChart(rows){
 const rate=rows.filter(r=>r.feature==='late_rate_60min_T2h');
 return lineChart(rate,'value','mean_predicted_risk',{title:'Predicted delay risk at different prior-hour late rates',caption:'Other inputs retain their observed values for the 1,500 sampled flights. The line connects the evaluated values.',xLabel:'Prior-hour late fraction (0–1)',yLabel:'Mean predicted risk (%)',minX:0,maxX:1,minY:0,maxY:1,percent:true});
}
async function optionalStudies(){
 const resp=await fetch('/data/studies.json');
 if(resp.status===404)return null;
 if(!resp.ok)throw Error('Could not load feature study');
 return resp.json();
}
async function render(){
if(path==='/'){
 const {summary}=await data('models'),best=summary.reduce((a,b)=>+a.ap>+b.ap?a:b);
 show('<div class="hero"><small>Average precision lift · '+names[best.model]+'</small><b>'+Number(best.lift).toFixed(2)+'×</b> the fold-specific late-rate baseline<p>Average precision '+dec(best.ap)+' &nbsp; / &nbsp; ROC-AUC '+dec(best.auc)+'</p></div>'+section('How the models compare','<p>Eleven models scored the same reporting flights. The five leading models span less than 0.005 ROC-AUC.</p><div class="links"><a href="/models">Compare ranking and spread across folds →</a><a href="/errors">See where the models missed late flights →</a><a href="/interpretation">Explore the airport-history features used by the trees →</a><a href="/explorer">Score a development flight →</a></div>'));
}else if(path==='/models'){
 const [m,tuning,curves]=await Promise.all([data('models'),data('tuning'),data('classification')]),r=m.summary.sort((a,b)=>b.ap-a.ap);
 show(section('XGBoost and LightGBM lead by a narrow margin','<p>Bar length shows average precision (0–0.5). Whiskers show one sample SD across six fold/seed scores.</p>'+bars(r,'model','ap','ap_sd',.5)+table(r.map(v=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,k==='model'?x:dec(x)]))),'Validation benchmark'))+
 section('Fold and seed details','<label>Model <select id="model">'+options(Object.keys(m.folds))+'</select></label><label>Curve period <select id="curvefold"><option>3</option><option>4</option></select></label><div id="details"></div>')+section('Depth, pruning and PCA',lineChart(m.curves.depth_curve,'max_depth','ap',{title:'Deeper candidate trees achieved higher tuning AP',caption:'Each point is the saved depth-level tuning summary. Compare candidate depths on folds 1–2.',xLabel:'Maximum tree depth',yLabel:'Average precision',minX:8,maxX:16,xTicks:[8,12,16],minY:.29,maxY:.32})+lineChart(m.curves.pca_variance,'components','cumulative_variance',{title:'Most input variance is concentrated in a fraction of the components',caption:'Cumulative training variance on fold 4. Retaining 95% selected 31 components; the benchmark table shows the effect on prediction.',xLabel:'Number of principal components',yLabel:'Cumulative variance (%)',minX:1,maxX:76,xTicks:[1,20,40,60,76],minY:0,maxY:1,percent:true})+Object.entries(m.curves).map(([k,v])=>'<details><summary>'+esc(k.replaceAll('_',' '))+'</summary>'+table(v,k)+'</details>').join('')));
 const update=()=>{const curve=curves[$('#model').value][$('#curvefold').value];const axes={minX:0,maxX:1,minY:0,maxY:1,percent:true,xTicks:[0,.25,.5,.75,1]};$('#details').innerHTML=lineChart(curve.roc,'false_positive_rate','recall',{...axes,diagonal:true,title:'How many delays are caught as false alarms increase?',caption:'ROC curve, seed 11 on the selected reporting fold. The diagonal shows chance ranking. Up to 130 threshold points are displayed.',xLabel:'False-positive rate (%)',yLabel:'Recall (%)'})+lineChart(curve.pr,'recall','precision',{...axes,baseline:curve.prevalence,title:'How precise are alerts at higher recall?',caption:'Precision–recall curve on the same flights. The dashed reference is the fold late rate. Each point corresponds to a different alert threshold.',xLabel:'Recall (%)',yLabel:'Precision (%)'})+table(m.folds[$('#model').value].map(({fold,seed,late_rate,ap,roc_auc})=>({fold,seed,late_rate:pct(late_rate),ap:dec(ap),roc_auc:dec(roc_auc),lift:(ap/late_rate).toFixed(2)})),'Fold scores')+'<details><summary>Search configurations and recorded runtimes · tuning folds 1–2</summary>'+table(tuning[$('#model').value],'Tuning runs')+'</details>';drawTables()};$('#model').onchange=$('#curvefold').onchange=update;update();
}else if(path==='/protocol'){
 const [audit,eda]=await Promise.all([data('protocol'),data('eda')]);
 show(section('A chronological holdout','<div class="flow">Jan–Aug training → Aug 31 gap → Sep–Nov validation → Nov 29–30 gap → December test</div><p>519,275 training flights, 207,704 validation flights and 68,146 test flights. Development uses expanding folds with a one-day gap. Folds 1–2 select settings.</p>')+section('Information available at the checkpoint','<div class="flow">Weather ≤ 180 min old &nbsp; + &nbsp; departures in (T−60 min, T] &nbsp; → &nbsp; T &nbsp; → &nbsp; scheduled departure at T+2h</div><p>Only earlier departures contribute observed delays. The current flight’s actual times and delay causes are excluded.</p>')+section('Development flight patterns',heatmap(eda.hour_weekday)+lineChart(eda.month,'month','late_rate',{title:'The share of late departures changes across months',caption:'Observed rates across the development exports. January–November coverage excludes protected dates and gaps. Hover for flight counts.',xLabel:'2025 local flight month',yLabel:'Late departures (%)',minX:1,maxX:11,xTicks:[1,3,5,7,9,11],minY:0,maxY:Math.ceil(Math.max(...eda.month.map(r=>r.late_rate))*10)/10,percent:true}))+section('Search the 35 inputs','<label>Feature or definition <input type=search id="find"></label><div id="audit"></div>'));
 content.insertAdjacentHTML('beforeend',section('Delay rates across the development cohort','<label>Compare by <select id="groupRate">'+options(['airport','carrier','hour','weekday','Visibility (miles)','Wind speed (knots)','Cloud base (feet)','Thunderstorm','Fog','Snow','Prior-hour late fraction','Prior-hour departures','Next-hour scheduled departures'])+'</select></label><div id="rates"></div>'));
 const updateRate=()=>{const group=$('#groupRate').value;const rows=eda[group].map(r=>{const label=Object.values(r)[0];return {category:group==='weekday'?['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][label]:label,flights:r.flights,late_rate:dec(r.late_rate)}});$('#rates').innerHTML='<p>Bars show the fraction delayed (0–1). The table gives the number of eligible flights.</p>'+bars(rows,'category','late_rate',null,1)+table(rows,'Development late rates');drawTables()};$('#groupRate').onchange=updateRate;updateRate();
 const filter=()=>{$('#audit').innerHTML=table(audit.filter(r=>Object.values(r).join(' ').toLowerCase().includes($('#find').value.toLowerCase())),'Column audit');drawTables()};$('#find').oninput=filter;filter();
}else if(path==='/features'){
 show(section('Which information improves delay prediction?',table([{Group:'A',Inputs:'Schedule and calendar',Count:7},{Group:'B',Inputs:'Origin, destination and carrier',Count:3},{Group:'C',Inputs:'Origin weather and report quality',Count:20},{Group:'D',Inputs:'Recent airport departures and scheduled demand',Count:5}],'Feature groups')));
 const study=await optionalStudies();
 if(study){
 content.insertAdjacentHTML('beforeend',section('What each feature group adds','<p>Locked XGBoost settings, seed 11. Compare validation AP on identical flights within each fold. Group gains depend on the order shown.</p><label>Airport <select id="studyAirport">'+options(['All','ATL','ORD','LAX'])+'</select></label><label>Fold <select id="studyFold">'+options(['3','4'])+'</select></label><div id="studyScores"></div>'));
 const update=()=>{
 const subset=study.scores.filter(r=>r.airport===$('#studyAirport').value && r.fold===$('#studyFold').value*1);
 const ladder=subset.filter(r=>r.hours===2).map((r,i,a)=>({group:r.group,flights:r.flights,ap:dec(r.ap),gain:i?dec(r.ap-a[i-1].ap):'—',roc_auc:dec(r.auc),lift:r.lift.toFixed(2)}));
 const horizons=subset.filter(r=>r.group==='ABCD').map(r=>({checkpoint:'T−'+r.hours+'h',flights:r.flights,ap:dec(r.ap),roc_auc:dec(r.auc),lift:r.lift.toFixed(2)}));
 $('#studyScores').innerHTML=bars(ladder,'group','ap',null,1)+table(ladder,'T-2h feature ladder')+section('Two hours versus six hours','<p>Both checkpoints use all four groups and the same settings. Each horizon uses its own weather and activity windows.</p>'+table(horizons,'Checkpoint comparison'));drawTables()};
 $('#studyAirport').onchange=$('#studyFold').onchange=update;update();}
}else if(path==='/interpretation'){
 const groups=await data('interpretation');show(section('Which inputs did each fitted model use most?','<p>Compare the estimator’s relative importance within one model. Correlated inputs can share importance. These values do not measure the gain from removing a feature.</p><label>Model <select id="which">'+options(Object.keys(groups))+'</select></label><div id="importance"></div>'));
 const update=()=>{const rows=groups[$('#which').value];$('#importance').innerHTML=bars(rows.slice(0,6),'feature','importance',null,Math.max(...rows.map(x=>+x.importance)))+table(rows,'Feature importance');drawTables()};$('#which').value='xgboost';$('#which').onchange=update;update();
 const study=await optionalStudies();
 if(study){
 const effects=study.effects.slice().sort((a,b)=>b.mean_abs_shap-a.mean_abs_shap);
 content.insertAdjacentHTML('beforeend',section('What influenced XGBoost on held-out flights?','<p>Fold 4, seed 11, fixed sample of 1,500 flights. Mean absolute TreeSHAP contribution measures movement in log-odds. Permutation AP drop measures the effect of shuffling one feature once; correlated features can mask each other.</p>'+bars(effects.slice(0,8),'feature','mean_abs_shap',null,effects[0].mean_abs_shap)+table(effects.map(r=>({feature:r.feature,mean_abs_shap:dec(r.mean_abs_shap),permutation_ap_drop:dec(r.permutation_ap_drop)})),'XGBoost held-out explanations'))+section('How predicted risk changes with one input','<p>Each row sets one feature to the indicated value for the same sample, then averages predictions. This can create unusual combinations of correlated inputs and describes model behaviour, not causal effects.</p>'+studyDependenceChart(study.dependence)+table(study.dependence.map(r=>({feature:r.feature,value:r.value,mean_predicted_risk:pct(r.mean_predicted_risk)})),'Partial dependence')));drawTables();}
}else if(path==='/errors'){
 const e=await data('errors');show('<div class="filters"><label>Model <select id="which">'+options(Object.keys(names))+'</select></label><label>Reporting fold <select id="fold"><option>3</option><option>4</option></select></label></div><div id="slices"></div>');
 function update(){const select=k=>e[k].filter(r=>r.model===$('#which').value&&r.fold===$('#fold').value).map(({model,fold,...r})=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,v===''?'—':isNaN(+v)||Number.isInteger(+v)?v:dec(v)])));
 $('#slices').innerHTML=section('Delay ranking by airport','<p>Bars show average precision by origin on a common 0–1 scale. The table includes each airport’s late rate and lift.</p>'+bars(select('robustness'),'airport','ap',null,1)+table(select('robustness'),'Airport performance'))+section('Missed delays by departure hour','<p>Each model flags the highest-risk 10% within the fold. Tied flights receive proportional credit. Seed 11.</p>'+bars(select('errors').filter(r=>r.hours!=='All'),'hours','recall',null,1)+'<p>Bar length is the fraction of late flights caught in each hour band (0–1).</p>'+table(select('errors'),'Alert outcomes'))+section('Predicted versus observed delay rates','<p>Compare the mean risk in each probability bin with the fraction actually delayed. Small bins are less stable.</p>'+lineChart(select('calibration').filter(r=>+r.flights>=100),'predicted','observed',{title:'Do predicted risks match observed delay rates?',caption:'Points above the dashed line indicate underestimated risk. Only bins with at least 100 flights are plotted; all bins appear in the table.',xLabel:'Mean predicted risk (%)',yLabel:'Observed late fraction (%)',minX:0,maxX:1,minY:0,maxY:1,percent:true,diagonal:true})+table(select('calibration'),'Calibration bins'));drawTables()}
 $('#which').value='xgboost';$('#which').onchange=$('#fold').onchange=update;update();
}else if(path==='/test'){
 const resp=await fetch('/data/test.json');if(resp.status===404){show('');return}if(!resp.ok)throw Error('Could not load the test summary');const test=await resp.json(),{summary}=await data('models');
 const rows=test.sort((a,b)=>b.average_precision-a.average_precision).map(r=>({model:r.model,development_ap:dec(summary.find(s=>s.model===r.model).ap),test_ap:dec(r.average_precision),test_auc:dec(r.roc_auc),test_lift:(r.average_precision/r.late_rate).toFixed(2)}));
 show('<div class="hero"><small>Final holdout · 1 December through 31 December</small><b>'+Number(test[0].flights).toLocaleString()+'</b> flights · '+pct(test[0].late_rate)+' left late</div>'+section('One evaluation of eleven frozen pipelines',table(rows,'December scores')));
}else if(path==='/explorer'){
 show('<p>Select an airport and date to find a scheduled departure. The XGBoost pipeline was refit on all development flights. These examples are in-sample.</p><form id="filters"><label>Origin <select id="airport">'+options(['ATL','ORD','LAX'])+'</select></label><label>Local flight date <input id="date" type="date" min="2025-09-01" max="2025-11-28" value="2025-10-13" required></label><button>Find flights</button></form><div id="choose"></div><div id="risk"></div>');
 let revision=0;async function flights(){const version=++revision;$('#choose').textContent='Loading flights…';$('#risk').textContent='';try{const list=await get('/api/flights?airport='+$('#airport').value+'&date='+$('#date').value);if(version!==revision)return;if(!list.length){$('#choose').textContent='No development flights match this date and airport.';return}
 $('#choose').innerHTML='<label>Flight <select id="flight">'+list.map(r=>'<option value="'+r.id+'">'+esc(r.label)+'</option>').join('')+'</select></label><button id="predict">Predict risk</button>';$('#flight').onchange=()=>{$('#risk').textContent=''};$('#predict').onclick=async()=>{const id=$('#flight').value;$('#risk').textContent='Scoring flight…';try{const r=await get('/api/predict?id='+id);if(version!==revision||id!==$('#flight').value)return;const date=new Date(r.date+'T12:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});$('#risk').innerHTML='<div class="hero"><small>'+esc(r.route)+' · '+date+' · record ID '+r.id+'</small><b>'+pct(r.p)+'</b> estimated probability of a departure delay of 15 minutes or more</div><p>September–November validation-export late rate for this airport and hour: '+pct(r.context)+'. Weather report age: '+(r.age===null?'no matched report':r.age+' min')+'. Prior mean delay: '+(r.history===null?'no eligible or unmasked history':r.history.toFixed(1)+' min')+'.</p><details><summary>Reveal observed outcome</summary><p>'+(r.actual?'Delayed at least 15 minutes':'Departed within the 15-minute threshold')+'</p></details>'}catch(err){$('#risk').textContent=err.message}};
 }catch(err){$('#choose').textContent=err.message}}$('#filters').onsubmit=e=>{e.preventDefault();flights()};await flights();
}else{
 show(section('Predicting at T−2h','<p>Each row describes an operated flight at Atlanta, Chicago O’Hare or Los Angeles. BTS schedules define the checkpoint. IEM weather is joined backwards by airport and UTC time. A 15-minute departure delay defines the positive class.</p>')+section('Checks and limitations','<p>The build includes a 300-row development check per horizon against direct airport-history recomputation. Archived weather observation time proxies for availability. Schedules are assumed known at T. Prior December departures may enter later history once observed.</p><p>One year, three hubs and two reporting folds limit generalisation. Earlier prototypes examined September–November. December’s higher late rate affects average precision.</p>')+section('Evidence','<div class="links"><a href="/protocol">Browse the input audit →</a><a href="/test">Read the frozen holdout results →</a><a href="https://github.com/harshith-raj-06/flight-delay-dataset">Project repository →</a></div>'));
}}
render().catch(err=>show(note(err.message+'. Refresh to retry.')));
