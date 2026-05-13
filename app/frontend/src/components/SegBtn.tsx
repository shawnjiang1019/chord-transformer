// src/components/SegBtn.tsx
// One button in a segmented control (verse / chorus / bridge).
// Wrap multiple SegBtn elements in a flex container with border-l-design on the wrapper.

import { type ReactNode } from 'react';

interface SegBtnProps {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}

export default function SegBtn({ active, onClick, children }: SegBtnProps) {
  return (
    <button
      onClick={onClick}
      className={[
        'flex-1 py-2 px-[6px] border-design border-l-0',
        'font-sans font-semibold text-xs tracking-[0.04em] uppercase cursor-pointer',
        active ? 'bg-ink text-white' : 'bg-white text-ink',
      ].join(' ')}
    >
      {children}
    </button>
  );
}
