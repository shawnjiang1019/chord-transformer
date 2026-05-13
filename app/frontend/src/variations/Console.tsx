'use client';

// src/variations/Console.tsx
// Variation A — 3-column layout: palette | prompt→generate→output | controls.

import { useState, useEffect, useRef } from 'react';
import Chip from '@/src/components/Chip';
import SectionLabel from '@/src/components/SectionLabel';
import Slider from '@/src/components/Slider';
import SegBtn from '@/src/components/SegBtn';
import PianoRoll from '@/src/components/PianoRoll';
import {
  type Chord,
  NOTES_SHARP,
  PALETTE,
  romansForKey,
  chordFromDegree,
  chordToRoman,
  chordLabel,
} from '@/src/engine/theory';
import { AudioEngine } from '@/src/engine/audio';
import { apiGenerate, apiBranch } from '@/src/engine/generate';

interface ConsoleProps {
  density?: 'compact' | 'roomy';
  showRoman?: boolean;
  showAdvanced?: boolean;
  accent?: string;
}

export default function Console({
  density = 'roomy',
  showRoman = true,
  showAdvanced = true,
  accent = '#ff3b00',
}: ConsoleProps) {
  const [prompt, setPrompt] = useState<Chord[]>([
    { root: 'C', q: 'maj' },
    { root: 'G', q: 'maj' },
    { root: 'A', q: 'min' },
  ]);
  const [output, setOutput] = useState<Chord[]>([]);
  const [branches, setBranches] = useState<Chord[][]>([]);
  const [locked, setLocked] = useState<number[]>([]);
  const [keyRoot, setKeyRoot] = useState('C');
  const [keyMode, setKeyMode] = useState('major');
  const [section, setSection] = useState('verse');
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(4);
  const [numChords, setNumChords] = useState(8);
  const [tempo, setTempo] = useState(96);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [playingIdx, setPlayingIdx] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [pianoRollHeight, setPianoRollHeight] = useState(162);

  const audioRef = useRef<AudioEngine | null>(null);
  const dragRef = useRef({ active: false, startY: 0, startH: 0 });

  useEffect(() => {
    audioRef.current = new AudioEngine();
    return () => { audioRef.current?.stop(); };
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragRef.current.active) return;
      const delta = dragRef.current.startY - e.clientY;
      setPianoRollHeight(h => Math.max(80, Math.min(420, dragRef.current.startH + delta)));
    };
    const onMouseUp = () => { dragRef.current.active = false; };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  const padY = density === 'compact' ? 'py-[14px]' : 'py-[22px]';
  const padX = density === 'compact' ? 'px-[18px]' : 'px-[28px]';
  const pad = `${padY} ${padX}`;

  // ── Actions ──────────────────────────────────────────────────────────

  const addToPrompt = (c: Chord) => setPrompt(p => [...p, c]);
  const removeFromPrompt = (i: number) => setPrompt(p => p.filter((_, j) => j !== i));

  const doGenerate = async () => {
    const opts = { seed: prompt, keyRoot, keyMode, section, temperature, topK, numChords };

    setGenerating(true);
    setGenerateError(null);
    setOutput([]);

    try {
      const [seq, newBranches] = await Promise.all([
        apiGenerate(opts),
        apiBranch(opts, 3),
      ]);

      // Re-apply any locked chords over the new sequence
      if (locked.length && output.length) {
        locked.forEach(i => {
          if (output[i]) seq[i] = { ...output[i], roman: chordToRoman(output[i], keyRoot, keyMode) };
        });
      }

      // Reveal animation
      const revealStep = (k: number) => {
        if (k > seq.length) {
          setBranches(newBranches);
          setGenerating(false);
          return;
        }
        setOutput(seq.slice(0, k));
        setTimeout(() => revealStep(k + 1), 90);
      };
      revealStep(1);

    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : 'Unknown error');
      setGenerating(false);
    }
  };

  const togglePlay = () => {
    const eng = audioRef.current;
    if (!eng) return;
    if (isPlaying) {
      eng.stop();
      setIsPlaying(false);
      setPlayingIdx(-1);
      return;
    }
    if (!output.length) return;
    setIsPlaying(true);
    eng.playSequence(
      output, tempo,
      i => setPlayingIdx(i),
      () => { setIsPlaying(false); setPlayingIdx(-1); },
    );
  };

  const toggleLock = (i: number) => {
    setLocked(L => L.includes(i) ? L.filter(x => x !== i) : [...L, i]);
  };

  const applyBranch = (b: Chord[]) => {
    setOutput(b);
    setBranches([]);
  };

  const canGenerate = prompt.length > 0 && !generating;

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div
      className="w-full h-screen bg-paper text-ink font-sans grid overflow-hidden"
      style={{ gridTemplateRows: `auto 1fr 8px ${pianoRollHeight}px` }}
    >

      {/* ── Top bar ───────────────────────────────────────────────── */}
      <div className={`flex items-center justify-between ${pad} border-b-design gap-5`}>

        {/* Branding */}
        <div>
          <div className="font-mono text-[10px] tracking-[0.22em] uppercase text-dim">
            v0.1 · console
          </div>
          <div className="font-sans font-bold text-[26px] tracking-[-0.02em] leading-none">
            CHORD<span style={{ color: accent }}>·</span>TRANSFORMER
          </div>
        </div>

        <div className="flex gap-4 items-center">
          {/* Key picker */}
          <div>
            <div className="font-mono text-[10px] text-dim tracking-[0.08em]">KEY</div>
            <div className="flex">
              <select
                value={keyRoot}
                onChange={e => setKeyRoot(e.target.value)}
                className="font-sans font-bold text-base px-2 py-1 border-design bg-white appearance-none"
              >
                {NOTES_SHARP.map(n => <option key={n} value={n}>{n}</option>)}
              </select>
              <select
                value={keyMode}
                onChange={e => setKeyMode(e.target.value)}
                className="font-sans font-semibold text-sm px-2 py-1 border-design border-l-0 bg-white appearance-none"
              >
                <option value="major">major</option>
                <option value="minor">minor</option>
              </select>
            </div>
          </div>

          {/* Tempo */}
          <div className="w-40">
            <Slider
              label="TEMPO"
              value={tempo}
              min={50} max={180} step={1}
              onChange={setTempo}
              format={v => `${v} BPM`}
              accent={accent}
            />
          </div>

          {/* Play / Stop */}
          <button
            onClick={togglePlay}
            disabled={!output.length}
            className="px-[18px] py-[10px] font-sans font-bold text-[13px] tracking-[0.1em] uppercase text-white"
            style={{
              background: isPlaying ? accent : '#0a0a0a',
              border: `1.4px solid ${isPlaying ? accent : '#0a0a0a'}`,
              cursor: output.length ? 'pointer' : 'not-allowed',
              opacity: output.length ? 1 : 0.4,
            }}
          >
            {isPlaying ? '■ STOP' : '▶ PLAY'}
          </button>
        </div>
      </div>

      {/* ── Main body: 3 columns ──────────────────────────────────── */}
      <div className="grid grid-cols-[260px_1fr_280px] overflow-hidden">

        {/* ── Left rail — palette ───────────────────────────────── */}
        <div className={`${pad} border-r-design overflow-y-auto`}>
          <SectionLabel num="01">CHORD PALETTE</SectionLabel>
          <div className="grid grid-cols-3 gap-2 mb-[18px]">
            {PALETTE.map((c, i) => (
              <div
                key={i}
                onClick={() => addToPrompt(c)}
                className="py-2 px-1 bg-white border-design font-sans font-semibold text-[13px] text-center cursor-pointer transition-colors duration-[80ms]"
                onMouseEnter={e => (e.currentTarget.style.background = accent + '15')}
                onMouseLeave={e => (e.currentTarget.style.background = '#ffffff')}
              >
                {chordLabel(c)}
              </div>
            ))}
          </div>

          <SectionLabel num="02">ROMAN NUMERALS</SectionLabel>
          <div className="font-mono text-[10px] text-dim mb-2">
            IN KEY OF {keyRoot} {keyMode.toUpperCase()}
          </div>
          <div className="grid grid-cols-4 gap-[6px] mb-[18px]">
            {romansForKey(keyMode).map((r, i) => {
              const ch = chordFromDegree(i + 1, keyRoot, keyMode);
              return (
                <div
                  key={r}
                  onClick={() => addToPrompt(ch)}
                  className="py-2 px-1 bg-chalk border-design font-mono font-medium text-xs text-center cursor-pointer"
                  onMouseEnter={e => (e.currentTarget.style.background = accent + '15')}
                  onMouseLeave={e => (e.currentTarget.style.background = '#fafaf6')}
                >
                  <div>{r}</div>
                  <div className="text-[9px] text-dim mt-[2px]">{chordLabel(ch)}</div>
                </div>
              );
            })}
          </div>

          <SectionLabel num="03">IMPORT</SectionLabel>
          <div className="py-[14px] px-[10px] border-design-dashed bg-chalk text-center font-mono text-[11px] text-dim cursor-pointer">
            <div className="font-semibold text-ink">+ MIDI / AUDIO</div>
            <div className="mt-1 text-[9.5px]">extract chords → seed prompt</div>
          </div>
          <div
            className="mt-[10px] py-[14px] px-[10px] bg-chalk text-center font-mono text-[11px] cursor-pointer"
            style={{ border: '1.4px dashed #a86a1f', color: '#a86a1f' }}
          >
            <div className="font-semibold">+ MELODY → HARMONIZER</div>
            <div className="mt-1 text-[9.5px] text-dim">chroma vectors for cross-attention</div>
          </div>
        </div>

        {/* ── Center — prompt / generate / output ───────────────── */}
        <div className={`${pad} overflow-y-auto bg-paper`}>

          {/* A — Prompt */}
          <SectionLabel num="A">PROMPT · YOUR CHORDS</SectionLabel>
          <div className="flex flex-wrap gap-[10px] p-[14px] min-h-20 border-design bg-chalk mb-[22px] items-center">
            {prompt.length === 0 && (
              <div className="font-mono text-xs text-dim">click palette chips to seed →</div>
            )}
            {prompt.map((c, i) => (
              <Chip
                key={`p-${i}`}
                chord={c}
                kind="prompt"
                onDelete={() => removeFromPrompt(i)}
                showRoman={showRoman}
                roman={chordToRoman(c, keyRoot, keyMode)}
                accent={accent}
              />
            ))}
          </div>

          {/* Generate button */}
          <div className="flex items-stretch mb-[22px]">
            <button
              onClick={() => doGenerate()}
              disabled={!canGenerate}
              className="flex-1 p-[18px] text-white font-sans font-bold text-lg tracking-[0.18em] uppercase border-design"
              style={{
                background: canGenerate ? '#0a0a0a' : '#cdc8b8',
                cursor: canGenerate ? 'pointer' : 'not-allowed',
              }}
            >
              {generating ? '▶ GENERATING…' : '▶ GENERATE  →'}
            </button>
            <button
              onClick={() => doGenerate()}
              disabled={!canGenerate}
              title="Re-sample"
              className="w-[60px] p-[18px] bg-white border-design border-l-0 font-mono text-base cursor-pointer"
            >
              ↻
            </button>
          </div>

          {/* B — Output */}
          <SectionLabel num="B">OUTPUT · GENERATED SEQUENCE</SectionLabel>
          {generateError && (
            <div className="mb-3 px-[14px] py-[10px] border-design font-mono text-[11px]" style={{ borderColor: accent, color: accent }}>
              ✕ {generateError}
            </div>
          )}
          <div className="flex flex-wrap gap-[10px] p-[14px] min-h-[110px] border-design bg-white mb-[22px] items-center">
            {output.length === 0 && !generating && (
              <div className="font-mono text-xs text-dim">
                output appears here · click lock badge to fix a chord
              </div>
            )}
            {output.map((c, i) => {
              const isPromptSlot = i < prompt.length;
              return (
                <Chip
                  key={`o-${i}`}
                  chord={c}
                  kind={isPromptSlot ? 'prompt' : (i === playingIdx ? 'sampled' : 'output')}
                  onClick={isPromptSlot ? undefined : () => toggleLock(i)}
                  locked={locked.includes(i)}
                  active={i === playingIdx}
                  showRoman={showRoman}
                  roman={c.roman ?? chordToRoman(c, keyRoot, keyMode)}
                  accent={accent}
                />
              );
            })}
            {generating && (
              <div
                className="p-3 min-w-16 font-mono text-xs text-center"
                style={{
                  border: `1.4px dashed ${accent}`,
                  color: accent,
                  animation: 'pulse 0.8s infinite',
                }}
              >
                ?
              </div>
            )}
          </div>

          {/* C — Branches */}
          {branches.length > 0 && (
            <>
              <SectionLabel num="C">ALTERNATIVE BRANCHES</SectionLabel>
              <div className="flex flex-col gap-[10px] mb-[22px]">
                {branches.map((b, bi) => (
                  <div key={bi} className="flex items-center gap-3 px-3 py-[10px] bg-white border-design">
                    <div className="font-mono text-[11px] bg-ink text-white px-[6px] py-[2px] shrink-0">
                      BR·{bi + 1}
                    </div>
                    <div className="flex gap-[6px] flex-wrap flex-1">
                      {b.map((c, i) => (
                        <span
                          key={i}
                          className={`font-sans font-semibold text-[13px] ${i < prompt.length ? 'text-dim' : 'text-ink'}`}
                        >
                          {chordLabel(c)}{i < b.length - 1 ? ' ·' : ''}
                        </span>
                      ))}
                    </div>
                    <button
                      onClick={() => applyBranch(b)}
                      className="px-[10px] py-[6px] bg-white text-ink border-design font-sans font-semibold text-[11px] tracking-[0.1em] uppercase cursor-pointer"
                      onMouseEnter={e => {
                        e.currentTarget.style.background = accent;
                        e.currentTarget.style.color = '#ffffff';
                        e.currentTarget.style.borderColor = accent;
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.background = '#ffffff';
                        e.currentTarget.style.color = '#0a0a0a';
                        e.currentTarget.style.borderColor = '#0a0a0a';
                      }}
                    >
                      → SWAP
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── Right rail — controls ─────────────────────────────── */}
        <div className={`${pad} border-l-design bg-sand overflow-y-auto`}>
          <SectionLabel num="04">SECTION</SectionLabel>
          <div className="flex mb-6 border-l-design">
            <SegBtn active={section === 'verse'}  onClick={() => setSection('verse')}>verse</SegBtn>
            <SegBtn active={section === 'chorus'} onClick={() => setSection('chorus')}>chorus</SegBtn>
            <SegBtn active={section === 'bridge'} onClick={() => setSection('bridge')}>bridge</SegBtn>
          </div>

          <Slider
            label="NUM CHORDS"
            value={numChords}
            min={4} max={16} step={1}
            onChange={setNumChords}
            accent={accent}
          />

          {showAdvanced && (
            <>
              <div className="mt-5 mb-[14px] pt-[14px] border-t-muted-dashed font-mono text-[10px] tracking-[0.18em] uppercase text-dim">
                SAMPLER
              </div>
              <Slider
                label="TEMPERATURE"
                value={temperature}
                min={0.2} max={2.0} step={0.05}
                onChange={setTemperature}
                format={v => v.toFixed(2)}
                accent={accent}
              />
              <Slider
                label="TOP-K"
                value={topK}
                min={1} max={10} step={1}
                onChange={setTopK}
                accent={accent}
              />

              {/* Forward pass readout */}
              <div className="mt-6 p-[14px] bg-chalk border-design font-mono text-[10px] text-ink leading-[1.6]">
                <div className="mb-1 text-dim tracking-[0.1em]">FORWARD PASS</div>
                <div>
                  seed_ids · ({prompt.length})<br />
                  + section · [{section}]<br />
                  + key · {keyRoot} {keyMode}<br />
                  → P(next | ·) / T<br />
                  → top_k({topK}) · multinomial
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Resize handle ─────────────────────────────────────────── */}
      <div
        className="border-t-design border-b-design bg-sand cursor-ns-resize flex items-center justify-center select-none"
        onMouseDown={e => {
          dragRef.current = { active: true, startY: e.clientY, startH: pianoRollHeight };
        }}
      >
        <div className="flex gap-[4px]">
          {[0, 1, 2, 3].map(i => <div key={i} className="w-5 h-px bg-dust" />)}
        </div>
      </div>

      {/* ── Bottom: piano roll ────────────────────────────────────── */}
      <div
        className={`bg-sand overflow-hidden`}
        style={{ padding: `${density === 'compact' ? 12 : 16}px ${density === 'compact' ? 18 : 28}px` }}
      >
        <PianoRoll chords={output} activeIdx={playingIdx} accent={accent} />
      </div>
    </div>
  );
}
