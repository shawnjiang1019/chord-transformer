// src/engine/generate.ts
// Chord generation: real API + Markov mock fallback.

import { type Chord, degreeFromChord, chordFromDegree, chordToRoman } from './theory';

export interface GenerateOptions {
  seed?: Chord[];
  keyRoot?: string;
  keyMode?: string;
  section?: string;
  temperature?: number;
  topK?: number;
  numChords?: number;
}

// Transition weights for each scale degree in major mode.
// Multiple entries = higher probability (sampled proportionally).
const TRANSITIONS_MAJOR: Record<number, number[]> = {
  1: [5, 6, 4, 2, 3, 4, 5],
  2: [5, 7, 4, 5, 5],
  3: [6, 4, 1],
  4: [5, 1, 6, 2, 5],
  5: [1, 6, 4, 1, 1, 6],
  6: [4, 2, 5, 1, 4, 4],
  7: [1, 3],
};

// Degree preferences that bias generation toward the character of each section.
const SECTION_BIAS: Record<string, { prefer?: number[] }> = {
  verse:  {},
  chorus: { prefer: [1, 4, 5, 6] },
  bridge: { prefer: [2, 6, 3] },
};

function sampleNext(current: number, temperature: number, topK: number, prefer?: number[]): number {
  const choices = TRANSITIONS_MAJOR[current] ?? [1, 4, 5];
  const counts: Record<number, number> = {};
  choices.forEach(d => { counts[d] = (counts[d] ?? 0) + 1; });
  prefer?.forEach(d => { counts[d] = (counts[d] ?? 0) + 1; });

  const T = Math.max(0.1, temperature);
  const weighted = Object.entries(counts)
    .map(([d, c]) => ({ d: +d, w: Math.pow(c, 1 / T) }))
    .sort((a, b) => b.w - a.w)
    .slice(0, Math.max(1, topK));

  const total = weighted.reduce((s, e) => s + e.w, 0);
  let r = Math.random() * total;
  for (const e of weighted) {
    r -= e.w;
    if (r <= 0) return e.d;
  }
  return weighted[0].d;
}

export function mockGenerate(opts: GenerateOptions): Chord[] {
  const {
    seed = [],
    keyRoot = 'C',
    keyMode = 'major',
    section = 'verse',
    temperature = 1.0,
    topK = 4,
    numChords = 8,
  } = opts;

  const bias = SECTION_BIAS[section] ?? {};
  const out: Chord[] = [...seed];
  let cur = seed.length
    ? degreeFromChord(seed[seed.length - 1], keyRoot, keyMode)
    : 1;

  while (out.length < numChords) {
    const next = sampleNext(cur, temperature, topK, bias.prefer);
    out.push(chordFromDegree(next, keyRoot, keyMode));
    cur = next;
  }

  return out.map(c => ({ ...c, roman: chordToRoman(c, keyRoot, keyMode) }));
}

export function mockBranch(opts: GenerateOptions, count = 3): Chord[][] {
  return Array.from({ length: count }, () => mockGenerate(opts));
}

// ── Real API ──────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

export async function apiGenerate(opts: GenerateOptions): Promise<Chord[]> {
  const res = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt:      (opts.seed ?? []).map(c => ({ root: c.root, q: c.q })),
      section:     opts.section     ?? 'verse',
      key_root:    opts.keyRoot     ?? 'C',
      key_mode:    opts.keyMode     ?? 'major',
      temperature: opts.temperature ?? 1.0,
      top_k:       opts.topK        ?? 4,
      num_chords:  opts.numChords   ?? 8,
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`Generate failed (${res.status}): ${detail}`);
  }

  return res.json() as Promise<Chord[]>;
}

export async function apiBranch(opts: GenerateOptions, count = 3): Promise<Chord[][]> {
  return Promise.all(Array.from({ length: count }, () => apiGenerate(opts)));
}
