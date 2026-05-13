// src/components/SectionLabel.tsx
// Numbered section header with ruled bottom border.
// Example: <SectionLabel num="01">CHORD PALETTE</SectionLabel>

import { type ReactNode } from 'react';

interface SectionLabelProps {
  children: ReactNode;
  num?: string;
}

export default function SectionLabel({ children, num }: SectionLabelProps) {
  return (
    <div className="flex items-baseline gap-[10px] mb-3 border-b-design pb-[6px]">
      {num && (
        <span className="font-mono text-[11px] bg-ink text-white px-[6px] py-[2px]">
          {num}
        </span>
      )}
      <span className="font-sans font-bold text-xs tracking-[0.16em] uppercase">
        {children}
      </span>
    </div>
  );
}
