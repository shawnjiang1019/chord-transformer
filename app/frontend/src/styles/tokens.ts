// src/styles/tokens.ts
// Design system constants — colors, typography, borders.

export const FONT_SANS = 'var(--font-space-grotesk), system-ui, sans-serif';
export const FONT_MONO = 'var(--font-jetbrains-mono), monospace';

export const C = {
  bg:      '#f4f1ea',
  text:    '#0a0a0a',
  muted:   '#7a7a72',
  rail:    '#ece8db',
  surface: '#fafaf6',
  white:   '#ffffff',
  mid:     '#cdc8b8',
  accent:  '#ff3b00',
  blue:    '#1e3a8a',
} as const;

export const BORDER = '1.4px solid #0a0a0a';
export const BORDER_DASHED = '1.4px dashed #0a0a0a';
