// src/components/Chip.tsx
// Single chord chip — prompt / output / sampled / palette variants.
// Border color and sampled background are dynamic (depend on accent prop) → inline style.

import { type Chord, chordLabel } from '@/src/engine/theory';

export type ChipKind = 'prompt' | 'output' | 'palette' | 'sampled';

interface ChipProps {
  chord: Chord;
  kind?: ChipKind;
  onClick?: () => void;
  onDelete?: () => void;
  locked?: boolean;
  active?: boolean;
  dim?: boolean;
  showRoman?: boolean;
  roman?: string;
  accent?: string;
}

export default function Chip({
  chord,
  kind = 'output',
  onClick,
  onDelete,
  locked = false,
  active = false,
  dim = false,
  showRoman = false,
  roman,
  accent = '#ff3b00',
}: ChipProps) {
  const isSampled = kind === 'sampled';

  return (
    <div
      onClick={onClick}
      className={[
        'relative min-w-16 text-center select-none font-sans font-semibold text-lg',
        'transition-[background,color] duration-[80ms]',
        isSampled ? 'text-white' : 'text-ink',
        !isSampled && 'bg-white',
        locked ? 'outline outline-2 -outline-offset-2 outline-denim' : '',
        dim ? 'opacity-35' : '',
        onClick ? 'cursor-pointer' : 'cursor-default',
      ].filter(Boolean).join(' ')}
      style={{
        padding: '12px 14px',
        border: `1.4px solid ${active ? accent : '#0a0a0a'}`,
        ...(isSampled && { background: accent }),
      }}
    >
      <div>{chordLabel(chord)}</div>

      {showRoman && roman && (
        <div className={[
          'font-mono font-normal text-[10px] mt-1 tracking-[0.05em]',
          isSampled ? 'text-white/85' : 'text-dim',
        ].join(' ')}>
          {roman}
        </div>
      )}

      {onDelete && (
        <span
          onClick={e => { e.stopPropagation(); onDelete(); }}
          className="absolute -top-2 -right-2 size-4 bg-ink text-white text-[10px] grid place-items-center cursor-pointer font-mono"
        >
          ×
        </span>
      )}

      {locked && (
        <span className="absolute -bottom-[6px] -left-[6px] size-[14px] bg-denim text-white text-[9px] grid place-items-center font-mono">
          L
        </span>
      )}
    </div>
  );
}
