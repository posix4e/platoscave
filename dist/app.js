const arms = {
 O:{label:'O / ORIGINAL',title:'Begin with the original SPC population.',description:'Keep the known validation and training source matches. This gives us the starting condition.',purpose:'What do we observe before removing either kind of shared source?',badge:'Starting condition',values:[3337,6261,9963]},
 C:{label:'C / VALIDATION-CLEAN',title:'Remove descriptions of validation sources.',description:'Exclude SPC paragraphs whose source images occur in COCO validation. The training-source matches remain.',purpose:'Compared with Rv, does excluding validation sources change more than removing the same number of rows at random?',badge:'Validation sources excluded',values:[0,6261,9963]},
 D:{label:'D / SOURCE-DISJOINT',title:'Then remove the known training-source matches.',description:'Starting from C, exclude SPC paragraphs whose source images occur in COCO training.',purpose:'Compared with Rp, does excluding these shared training sources change more than the number of rows alone?',badge:'Known cross-source matches excluded',values:[0,0,9963]},
 Rv:{label:'Rv / RANDOM → C SIZE',title:'Remove as many rows, chosen at random.',description:'Choose a random subset of the original SPC population with the same 16,224 rows as C. Keep descriptions of the same source together.',purpose:'How much changes merely because fewer rows remain? Their class mix and caption style may still differ from C.',badge:'Validation overlap remains',values:[2772,5119,8333]},
 Rp:{label:'Rp / RANDOM → D SIZE',title:'Keep as many rows as D, chosen from C.',description:'Choose a random subset of C with the same 9,963 rows as D. Keep descriptions of the same source together.',purpose:'What follows from reducing the row count from C to D, while still excluding known validation sources?',badge:'Training overlap remains',values:[0,3777,6186]}
};
const format = n => n.toLocaleString('en-US');
const labels=['val','train','other','drop'];
for(const button of document.querySelectorAll('[data-arm]')){
 button.addEventListener('click',()=>{
  const arm=arms[button.dataset.arm];
  for(const other of document.querySelectorAll('[data-arm]')) other.setAttribute('aria-pressed',String(other===button));
  for(const key of ['label','title','description','purpose','badge'])document.getElementById('arm-'+key).textContent=arm[key];
  const count=arm.values.reduce((a,b)=>a+b,0);
  document.getElementById('arm-count').textContent=format(count);
  const values=[...arm.values,19561-count];
  labels.forEach((key,i)=>{document.getElementById('bar-'+key).style.width=(100*values[i]/19561)+'%';document.getElementById('count-'+key).textContent=format(values[i]);});
  document.getElementById('stack-bar').setAttribute('aria-label',`${arm.label}: ${format(values[0])} validation-source rows, ${format(values[1])} training-source rows, ${format(values[2])} rows with no known COCO match, and ${format(values[3])} removed rows.`);
 });
}
fetch('downloads/pilot-metrics.json').then(response=>{if(!response.ok)throw new Error('Unavailable');return response.json();}).then(data=>{
 const body=document.getElementById('run-rows');
 for(const run of data.runs){
  const row=document.createElement('tr');
  for(const value of [run.arm,run.fit_seed,'Complete',run.metrics.foscttm.toFixed(4),run.fit_seconds.toFixed(2)+' s']){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
  body.append(row);
 }
}).catch(()=>{const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=5;cell.textContent='The run table could not be loaded. Download the metric record below.';row.append(cell);document.getElementById('run-rows').append(row);});
