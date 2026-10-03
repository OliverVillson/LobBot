(() => {
  'use strict';
  const shape = '<g fill="none" stroke="currentColor" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"><path d="M53 15C35 9 18 18 21 34C6 37 5 59 20 64C8 77 18 94 34 92C33 109 53 120 65 106C81 119 100 106 97 92C112 95 125 76 113 65C125 53 116 32 102 35C104 20 89 9 74 15"/><path d="M42 37C61 34 63 49 51 56C40 63 39 80 53 86M84 39C73 41 72 56 84 62C97 68 94 83 81 86"/></g>';
  document.querySelectorAll('.sillon').forEach(svg => { svg.innerHTML = shape; });
  const get = id => document.getElementById(id);
  let palettes = [], current, busy = false;
  const query = new URLSearchParams(location.search);
  const roleNames = {canvas:'Fond',surface:'Surface',ink:'Encre',secondary:'Texte secondaire',accent:'Accent'};

  function choose(palette) {
    current = palette;
    document.body.dataset.palette = palette.id;
    document.querySelector('meta[name="theme-color"]').content = palette.canvas;
    document.querySelectorAll('.palette-option').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.id === palette.id)));
    get('palette-title').textContent = palette.name;
    get('palette-label').textContent = palette.name;
    const [description, note] = palette.note.split(' · ');
    get('palette-description').textContent = description;
    get('palette-note').textContent = note.charAt(0).toUpperCase() + note.slice(1) + '.';
    get('download-logo').href = `assets/palettes/sillon-${palette.id}-accent.svg`;
    get('download-tokens').href = `assets/palettes/tokens-${palette.id}.json`;
    get('color-swatches').replaceChildren(...Object.entries(roleNames).map(([role, name]) => {
      const swatch = document.createElement('div');
      const color = document.createElement('div'); color.className = 'swatch-color'; color.style.background = palette[role];
      const meta = document.createElement('div'); meta.className = 'swatch-meta';
      const label = document.createElement('strong'); label.textContent = name;
      const value = document.createElement('code'); value.textContent = palette[role];
      meta.append(label, value); swatch.append(color, meta); return swatch;
    }));
    const url = new URL(location.href); url.searchParams.set('palette', palette.id); history.replaceState(null, '', url);
  }

  function status(kind, label, description, width) {
    get('status').className = `status ${kind}`;
    get('status').textContent = label;
    get('result-description').textContent = description;
    get('candidate').style.width = width;
  }

  get('budget').addEventListener('input', event => { get('budget-value').textContent = `${event.target.value} Go`; });
  get('capability').addEventListener('change', event => {
    get('capability-name').textContent = event.target.options[event.target.selectedIndex].text.toLowerCase();
    status('neutral', 'Prêt à tester', 'La capacité a changé. Lancez une nouvelle expérience visuelle.', '100%');
  });
  get('run').addEventListener('click', () => {
    if (busy) return;
    busy = true; get('run').disabled = true; get('restore').disabled = true; get('capability').disabled = true;
    status('neutral', 'Évaluation…', 'Simulation : une modification est proposée, puis évaluée sur la capacité choisie.', '72%');
    setTimeout(() => {
      status('success', 'Capacité conservée', 'État de réussite : la modification satisfait le critère et peut être conservée.', '61%');
      busy = false; get('run').disabled = false; get('restore').disabled = false; get('capability').disabled = false;
    }, 800);
  });
  get('restore').addEventListener('click', () => { status('restored', 'Modification restaurée', 'État de restauration : le critère n’est pas satisfait. La modification est annulée.', '100%'); });
  fetch('assets/palettes/palettes.json').then(response => {
    if (!response.ok) throw new Error('Palette data unavailable');
    return response.json();
  }).then(data => {
    palettes = data;
    palettes.forEach(palette => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'palette-option'; button.dataset.id = palette.id; button.setAttribute('aria-pressed', 'false');
      const dots = document.createElement('span'); dots.className = 'palette-dots'; dots.setAttribute('aria-hidden', 'true');
      [palette.ink, palette.accent].forEach(color => { const dot = document.createElement('i'); dot.style.background = color; dots.append(dot); });
      button.append(dots, document.createTextNode(palette.name)); button.addEventListener('click', () => choose(palette)); get('palette-picker').append(button);
    });
    choose(palettes.find(palette => palette.id === query.get('palette')) || palettes[0]);
  }).catch(() => {
    const note = document.createElement('p'); note.textContent = 'Les palettes n’ont pas pu être chargées. Rechargez la page pour réessayer.'; get('palette-picker').append(note);
  });
})();
