// src/engine/theory.ts
// Music theory helpers — ported from prototype/engine.js.

export interface Chord {
  root: string;
  q: string;
  roman?: string;
}

export const NOTES_SHARP = [
  'C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
] as const;

export const NOTES_FLAT = [
  'C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B',
] as const;

export const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
export const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10];
export const MAJOR_ROMANS = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'];
export const MINOR_ROMANS = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'];

export const QUALITIES: Record<string, number[]> = {
  maj:  [0, 4, 7],
  min:  [0, 3, 7],
  dim:  [0, 3, 6],
  aug:  [0, 4, 8],
  '7':  [0, 4, 7, 10],
  maj7: [0, 4, 7, 11],
  min7: [0, 3, 7, 10],
  sus2: [0, 2, 7],
  sus4: [0, 5, 7],
  '6':  [0, 4, 7, 9],
  add9: [0, 4, 7, 14],
};

export const Q_LABEL: Record<string, string> = {
  maj:  '',
  min:  'm',
  dim:  '°',
  aug:  '+',
  '7':  '7',
  maj7: 'maj7',
  min7: 'm7',
  sus2: 'sus2',
  sus4: 'sus4',
  '6':  '6',
  add9: 'add9',
};

export const PALETTE: Chord[] = [
  { root: 'C',  q: 'maj'  }, { root: 'D',  q: 'min'  }, { root: 'E',  q: 'min'  },
  { root: 'F',  q: 'maj'  }, { root: 'G',  q: 'maj'  }, { root: 'A',  q: 'min'  },
  { root: 'A',  q: 'maj'  }, { root: 'D',  q: 'maj'  }, { root: 'E',  q: 'maj'  },
  { root: 'B',  q: 'min'  }, { root: 'C',  q: 'maj7' }, { root: 'F',  q: 'maj7' },
  { root: 'G',  q: '7'    }, { root: 'D',  q: 'min7' }, { root: 'A',  q: 'min7' },
  { root: 'E',  q: 'min7' }, { root: 'C',  q: 'sus2' }, { root: 'G',  q: 'sus4' },
  { root: 'B',  q: 'dim'  }, { root: 'A',  q: '7'    },
];

export function noteIndex(name: string): number {
  const i = (NOTES_SHARP as readonly string[]).indexOf(name);
  if (i >= 0) return i;
  return (NOTES_FLAT as readonly string[]).indexOf(name);
}

export function chordLabel(c: Chord): string {
  return c.root + (Q_LABEL[c.q] ?? c.q);
}

export function chordMidi(c: Chord, bass = true): number[] {
  const root = noteIndex(c.root);
  const intervals = QUALITIES[c.q] ?? QUALITIES.maj;
  const voicing = intervals.map(i => 60 + root + i);
  return bass ? [48 + root, ...voicing] : voicing;
}

export function chordToRoman(c: Chord, keyRoot: string, keyMode = 'major'): string {
  const scale = keyMode === 'major' ? MAJOR_SCALE : MINOR_SCALE;
  const romans = keyMode === 'major' ? MAJOR_ROMANS : MINOR_ROMANS;
  const k = noteIndex(keyRoot);
  const r = noteIndex(c.root);
  const degreeSemis = ((r - k) % 12 + 12) % 12;
  const degree = scale.indexOf(degreeSemis);
  if (degree < 0) return '♭' + (Math.floor(degreeSemis / 2) + 1);
  let label = romans[degree];
  const isMinor = c.q.startsWith('min') || c.q === 'dim';
  const isDim = c.q === 'dim';
  label = isMinor
    ? label.toLowerCase().replace('°', '')
    : label.toUpperCase().replace('°', '');
  if (isDim) label += '°';
  else if (c.q === '7') label += '⁷';
  else if (c.q === 'maj7') label += 'M⁷';
  else if (c.q === 'min7') label = label.toLowerCase() + '⁷';
  else if (c.q === 'sus2' || c.q === 'sus4') label += 'sus';
  return label;
}

export function romanToChord(roman: string, keyRoot: string, keyMode = 'major'): Chord {
  const scale = keyMode === 'major' ? MAJOR_SCALE : MINOR_SCALE;
  const k = noteIndex(keyRoot);
  const m = roman.match(/^([b#♭♯]?)([ivIV]+)(°|7|M7|sus)?$/);
  if (!m) return { root: keyRoot, q: 'maj' };
  const [, acc, numeral, suffix = ''] = m;
  const numMap: Record<string, number> = { I:1, II:2, III:3, IV:4, V:5, VI:6, VII:7 };
  const degree = numMap[numeral.toUpperCase()] - 1;
  let semis = scale[degree] ?? 0;
  if (acc === 'b' || acc === '♭') semis -= 1;
  if (acc === '#' || acc === '♯') semis += 1;
  const root = NOTES_SHARP[(k + semis + 12) % 12];
  const isLower = numeral === numeral.toLowerCase();
  let q = isLower ? 'min' : 'maj';
  if (suffix === '°') q = 'dim';
  else if (suffix === '7') q = isLower ? 'min7' : '7';
  else if (suffix === 'M7') q = 'maj7';
  else if (suffix === 'sus') q = 'sus4';
  return { root, q };
}

export function romansForKey(keyMode = 'major'): string[] {
  return keyMode === 'major' ? [...MAJOR_ROMANS] : [...MINOR_ROMANS];
}

export function chordFromDegree(degree: number, keyRoot: string, keyMode = 'major'): Chord {
  const scale = keyMode === 'major' ? MAJOR_SCALE : MINOR_SCALE;
  const k = noteIndex(keyRoot);
  const semis = scale[degree - 1] ?? 0;
  const root = NOTES_SHARP[(k + semis + 12) % 12];
  const QUAL_MAJOR = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim'];
  const QUAL_MINOR = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj'];
  const q = (keyMode === 'major' ? QUAL_MAJOR : QUAL_MINOR)[degree - 1] ?? 'maj';
  return { root, q };
}

export function degreeFromChord(c: Chord, keyRoot: string, keyMode = 'major'): number {
  const scale = keyMode === 'major' ? MAJOR_SCALE : MINOR_SCALE;
  const k = noteIndex(keyRoot);
  const r = noteIndex(c.root);
  const d = ((r - k) % 12 + 12) % 12;
  const idx = scale.indexOf(d);
  return idx >= 0 ? idx + 1 : 1;
}
