// src/components/PianoRoll.tsx
// Horizontal piano-roll visualization of a chord sequence.
// Note positions are computed and require inline styles; wrapper uses Tailwind.

import { type Chord, chordMidi } from '@/src/engine/theory';

interface PianoRollProps {
  chords: Chord[];
  activeIdx: number;
  accent?: string;
}

export default function PianoRoll({ chords, activeIdx, accent = '#ff3b00' }: PianoRollProps) {
  if (!chords.length) {
    return (
      <div className="h-[130px] border-design bg-chalk grid place-items-center text-dim font-mono text-[11px]">
        piano roll · generate to populate
      </div>
    );
  }

  const allNotes = chords.flatMap(c => chordMidi(c));
  const lo = Math.min(...allNotes) - 1;
  const hi = Math.max(...allNotes) + 1;
  const rowH = 8;
  const cellW = 100 / chords.length;
  const height = (hi - lo + 1) * rowH + 28;

  return (
    <div className="border-design bg-chalk p-2">
      {/* Header */}
      <div className="flex justify-between font-mono text-[10px] text-dim mb-1 tracking-[0.08em] uppercase">
        <span>piano roll</span>
        <span>{chords.length} steps · 2 beats / chord</span>
      </div>

      {/* Grid */}
      <div
        className="relative bg-white"
        style={{ height, border: '1px solid #d8d4c5' }}
      >
        {/* Column gridlines */}
        {chords.map((_, i) => (
          <div
            key={`gl-${i}`}
            className="absolute top-0 bottom-0 w-px bg-[#e8e3d4]"
            style={{ left: `${(i + 1) * cellW}%` }}
          />
        ))}

        {/* Note blocks */}
        {chords.map((c, i) =>
          chordMidi(c).map((n, j) => (
            <div
              key={`n-${i}-${j}`}
              className="absolute"
              style={{
                left: `${i * cellW + 0.5}%`,
                width: `${cellW - 1}%`,
                top: (hi - n) * rowH,
                height: rowH - 1,
                background: i === activeIdx ? accent : (j === 0 ? '#1e3a8a' : '#0a0a0a'),
                opacity: j === 0 ? 1 : 0.85,
              }}
            />
          ))
        )}

        {/* Active column overlay */}
        {activeIdx >= 0 && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none"
            style={{
              left: `${activeIdx * cellW}%`,
              width: `${cellW}%`,
              background: `${accent}0f`,
              borderLeft: `2px solid ${accent}`,
              borderRight: `2px solid ${accent}`,
            }}
          />
        )}
      </div>
    </div>
  );
}
