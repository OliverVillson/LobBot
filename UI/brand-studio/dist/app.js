const directions={
  incision:{name:'Incision',title:'Le geste devient le signe.',description:'Un B massif traversé par une incision oblique. La matière se sépare, le signe reste lisible. LOBBOT affirme le droit de retirer, avec la précision d’un instrument.',risk:'La piste la plus mémorable et la plus provocante. La référence chirurgicale demande un discours rigoureux, sans gore ni promesse de résultat garanti.',tagline:'Cut the model.<br>Keep the capability.',caption:'Une incision. Une identité.',symbolCaption:'Retirer. Tester. Préserver.',alt:'Logo Incision : un B traversé par une coupe oblique',accent:['Incision','#F04B32'],ink:['Carbone','#191B18'],paper:'#F5F5F1',muted:['Acier','#C9CDC5'],font:'Archivo',fontRole:'Structure & titres',palette:'Le vermillon marque la zone d’intervention. Le carbone porte les informations. Le fond clair facilite la lecture du code et des schémas.',languageTitle:'Une coupe comme grammaire.',language:'Plans séparés, interruptions obliques, aplats nets, traits de repérage. Les fragments superflus peuvent sortir du cadre ; le noyau de capacité reste toujours ancré.'},
  core:{name:'Noyau',title:'La fonction au centre de tout.',description:'Des couches ouvertes autour d’un carré intact. Chaque enveloppe peut disparaître, mais le centre reste fixe. Le signe parle de protection et de structure, sans passer par l’imagerie médicale.',risk:'La piste la plus sereine pour une infrastructure. Sa géométrie peut évoquer une puce : l’ouverture asymétrique et le noyau isolé doivent rester distinctifs.',tagline:'Less model.<br>Same mission.',caption:'Tout converge vers l’essentiel.',symbolCaption:'Le noyau reste. Les couches cèdent.',alt:'Logo Noyau : des couches carrées ouvertes autour d’un noyau préservé',accent:['Cœur vif','#D6E95A'],ink:['Olive dense','#293326'],paper:'#F3F5ED',muted:['Membrane','#BAC4AE'],font:'Manrope',fontRole:'Clarté & humanité',palette:'Un jaune acide donne une présence singulière. L’olive profond apporte du calme. La couleur signale ce qui est conservé, plutôt que ce qui est retiré.',languageTitle:'La protection comme grammaire.',language:'Enveloppes ouvertes, cadrages concentriques, centres fixes et marges généreuses. La densité diminue depuis l’extérieur ; la fonction demeure au centre.'},
  signal:{name:'Signal',title:'Retirer le bruit. Garder le signal.',description:'Une matrice de barres se réduit autour d’un signal de longueur constante. La marque fait voir une capacité qui traverse l’intervention. Le logo se lit comme une mesure, jamais comme un cerveau.',risk:'La piste la plus technique et la plus expérimentale. Les fines barres demandent une version simplifiée pour les favicons, la gravure et les très petites tailles.',tagline:'Intelligence,<br>reduced to purpose.',caption:'Une fonction. Un signal continu.',symbolCaption:'Moins de structure. Le signal persiste.',alt:'Logo Signal : une matrice de barres qui protège un signal central',accent:['Trace','#FF894C'],ink:['Graphite','#212321'],paper:'#EFF1EF',muted:['Bruit','#BFC5BF'],font:'Barlow Semi Condensed',fontRole:'Densité & tension',palette:'L’orange trace la fonction au milieu du graphite. Le vert-gris pâle sert de fond de lecture. La palette conserve sa force en monochrome.',languageTitle:'La mesure comme grammaire.',language:'Champs de barres, matrices parcimonieuses, longueurs comparables et lignes de référence. Les éléments restent à leur place quand ils disparaissent, pour rendre la réduction lisible.'}
};
let activeDirection='incision';
function setText(id,text){const el=document.getElementById(id);if(el)el.textContent=text;}
function selectDirection(key,{scroll=false}={}){
  if(!directions[key])return;
  activeDirection=key;const d=directions[key];document.body.dataset.direction=key;
  document.querySelectorAll('[data-select]').forEach(button=>{if(button.hasAttribute('aria-pressed'))button.setAttribute('aria-pressed',String(button.dataset.select===key));});
  const symbol=document.getElementById('hero-symbol');if(symbol){symbol.src=`assets/logos/${key}-symbol-ink.svg`;symbol.alt=d.alt;}
  const wordmark=document.getElementById('hero-wordmark');if(wordmark)wordmark.src=`assets/logos/${key}-wordmark-ink.svg`;
  const mini=document.getElementById('mini-lockup');if(mini)mini.src=`assets/logos/${key}-lockup-white.svg`;
  const tagline=document.getElementById('brand-tagline');if(tagline)tagline.innerHTML=d.tagline;
  setText('specimen-label',`LOBBOT / ${d.name.toUpperCase()}`);setText('wordmark-caption',d.caption);setText('symbol-caption',d.symbolCaption);setText('direction-title',d.title);setText('direction-description',d.description);setText('direction-risk',d.risk);setText('accent-name',d.accent[0]);setText('accent-hex',d.accent[1]);setText('ink-name',d.ink[0]);setText('ink-hex',d.ink[1]);setText('paper-hex',d.paper);setText('muted-name',d.muted[0]);setText('muted-hex',d.muted[1]);setText('font-name',d.font);setText('font-role',d.fontRole);setText('palette-description',d.palette);setText('language-title',d.languageTitle);setText('language-description',d.language);
  const download=document.getElementById('download-logo');if(download)download.href=`assets/logos/${key}-lockup-ink.svg`;
  document.querySelectorAll('.landing-link').forEach(a=>a.href=`landing.html?direction=${key}`);
  document.querySelectorAll('.identity-link').forEach(a=>a.href=`index.html?direction=${key}`);
  const headerSymbol=document.querySelector('.brand img');if(headerSymbol)headerSymbol.src=`assets/logos/${key}-symbol-ink.svg`;
  const query=new URLSearchParams(location.search);query.set('direction',key);history.replaceState(null,'',`${location.pathname}?${query}${location.hash}`);
  if(scroll){document.getElementById('directions')?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
}
document.querySelectorAll('[data-select]').forEach(button=>button.addEventListener('click',()=>selectDirection(button.dataset.select,{scroll:button.classList.contains('gallery-item')})));
selectDirection(new URLSearchParams(location.search).get('direction')||'incision');

document.querySelectorAll('[data-experiment]').forEach(experiment=>{
  const model=experiment.querySelector('[data-model]');const state=experiment.querySelector('[data-state]');const description=experiment.querySelector('[data-description]');const run=experiment.querySelector('[data-run]');const reset=experiment.querySelector('[data-reset]');
  const english=document.documentElement.lang==='en';
  const messages=english?{
    initial:'Original model',intro:'The protected signal stays fixed. Surrounding components can be removed.',modify:'Modify components',test:'Test target capability',keep:'Keep the change',restore:'Restore the change',done:'Specialized structure',modifyDesc:'A candidate group of components is removed. The target capability is evaluated next.',testDesc:'The target capability is evaluated against the acceptance criterion.',keepDesc:'This illustrative test passes. The agent keeps the smaller structure.',restoreDesc:'This illustrative test fails. The previous components are restored.',doneDesc:'The example ends with a smaller structure and the protected signal still present.',run:'Run again'
  }:{
    initial:'Modèle initial',intro:'Le signal reste ancré. Les composants autour peuvent être retirés.',modify:'Modifier les composants',test:'Tester la capacité',keep:'Conserver le changement',restore:'Restaurer le changement',done:'Structure spécialisée',modifyDesc:'Un groupe de composants est retiré. La capacité ciblée va être évaluée.',testDesc:'La capacité ciblée est évaluée selon le critère d’acceptation.',keepDesc:'Ce test illustratif passe. L’agent conserve la structure réduite.',restoreDesc:'Ce test illustratif échoue. Les composants précédents sont restaurés.',doneDesc:'L’exemple se termine avec une structure réduite et le signal protégé toujours présent.',run:'Relancer l’expérience'
  };
  const heights=[72,111,89,135,119,152,102,124,154,136,120,154,141,155,120,134,100,151,119,132,84,110,74,54];
  heights.forEach((height,i)=>{const bar=document.createElement('span');bar.className=`model-bar${i>=9&&i<=14?' protected':''}`;bar.style.setProperty('--height',`${height}px`);bar.setAttribute('aria-hidden','true');model.append(bar);});
  const bars=[...model.children];let generation=0;let isRunning=false;
  const originalLabel=run.innerHTML;
  const setState=(key)=>{state.textContent=messages[key];description.textContent=messages[`${key}Desc`]||messages.intro;};
  const delay=()=>new Promise(resolve=>setTimeout(resolve,matchMedia('(prefers-reduced-motion: reduce)').matches?120:820));
  async function stage(key,indices,className,token){if(token!==generation)return false;setState(key);if(className)indices.forEach(i=>bars[i].classList.add(className));await delay();return token===generation;}
  async function start(){if(isRunning)return;generation++;const token=generation;isRunning=true;run.disabled=true;bars.forEach(bar=>bar.classList.remove('removed','removing','failed'));const first=[0,1,2,3,20,21,22,23];
    if(!await stage('modify',first,'removing',token))return;
    if(!await stage('test',[],null,token))return;
    first.forEach(i=>{bars[i].classList.remove('removing');bars[i].classList.add('removed');});
    if(!await stage('keep',[],null,token))return;
    const risky=[8,15];
    if(!await stage('modify',risky,'removing',token))return;
    if(!await stage('test',risky,'failed',token))return;
    setState('restore');risky.forEach(i=>bars[i].classList.remove('removing','failed'));
    await delay();if(token!==generation)return;
    const second=[4,5,18,19];
    if(!await stage('modify',second,'removing',token))return;
    if(!await stage('test',[],null,token))return;
    second.forEach(i=>{bars[i].classList.remove('removing');bars[i].classList.add('removed');});
    if(!await stage('keep',[],null,token))return;
    setState('done');isRunning=false;run.disabled=false;run.innerHTML=`${messages.run}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 10 7-10 7z"/></svg>`;
  }
  function clear(){generation++;isRunning=false;run.disabled=false;run.innerHTML=originalLabel;bars.forEach(bar=>bar.classList.remove('removed','removing','failed'));setState('initial');}
  state.setAttribute('role','status');state.setAttribute('aria-live','polite');run.addEventListener('click',start);reset.addEventListener('click',clear);
});
