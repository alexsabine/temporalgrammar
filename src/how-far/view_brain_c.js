const bind = (id, fn) => { const e = $(id); e.addEventListener('input', () => { fn(+e.value); kick(); }); fn(+e.value); };
let labTimer = 0, ioTimer = 0;
bind('vox', v => { const mm = VOX[v]; $('voxOut').textContent = M.fmtLen(mm * 1e-3, 2); st.vox = mm; if (MRI && MRI.vox !== mm) MRI.voxelise(mm); dirty = true; if (st.level === 0) { facts(); say(); } });
bind('lab', v => { $('labOut').textContent = fmt(v, 1) + '%'; if (Math.abs(labelFrac - v / 100) < 1e-9) return; labelFrac = v / 100; clearTimeout(labTimer); labTimer = setTimeout(() => { rebake(1); kick(); }, 140); if (st.level === 1) say(); });
function pickSite(k) { st.site = k; const T = M.NEURON, pi = M.PATH.indexOf(k); if (pi > 0) $('site').value = pi; $('siteOut').textContent = fmt(T.dist[k], 0) + ' µm'; D3.reset(); TR.n = 0; TR.site.fill(0); TR.soma.fill(0); clearTimeout(ioTimer); if (st.level === 2) ioTimer = setTimeout(() => { computeIO(); if (st.level === 2) { say(); maths(); } kick(); }, 120); else io3 = null; if (st.level === 2) { facts(); maths(); } dirty = true; kick(); }
$('site').max = M.PATH.length - 1; $('site').value = M.PATH.indexOf(M.HOME_SITE);
bind('site', v => { if (M.PATH[v] !== st.site || $('siteOut').textContent === '') pickSite(M.PATH[v]); });
bind('nsyn', v => { st.nsyn = v; $('nsynOut').textContent = String(v); rigDirty = true; if (st.level === 2) say(); });
$('fire3').addEventListener('click', () => { st.touched = true; fire3(); kick(); });
$('nmda').addEventListener('click', () => { st.nmda = !st.nmda; $('nmda').setAttribute('aria-pressed', String(st.nmda)); rigDirty = true; say(); kick(); });
bind('recon', v => { st.recon = v / 100; $('reconOut').textContent = v + '%'; dirty = true; });
bind('prel', v => { st.p = v; $('prelOut').textContent = fmt(v, 2); resetQ(); if (st.level === 4) { say(); maths(); } });
bind('nrel', v => { st.nrel = v; $('nrelOut').textContent = String(v); resetQ(); if (st.level === 4) { say(); facts(); maths(); } dirty = true; });
$('fire5').addEventListener('click', () => { fire5(performance.now()); maths(); kick(); });
$('auto5').addEventListener('click', () => { st.auto5 = !st.auto5; $('auto5').setAttribute('aria-pressed', String(st.auto5)); kick(); });
$('many5').addEventListener('click', () => { const now = performance.now(); for (let k = 0; k < 500; k++) fire5(now); say(); maths(); kick(); });
$('clear5').addEventListener('click', () => { resetQ(); say(); maths(); kick(); });
bind('volt', v => { st.V = v; $('voltOut').textContent = fmt(v, 0) + ' mV'; CH.setV(v); rigDirty = true; if (st.level >= 5) { say(); facts(); maths(); } });
bind('nch', v => { st.nch = NCH[v]; $('nchOut').textContent = String(st.nch); resetCH(); if (st.level === 5) { say(); facts(); maths(); } });
