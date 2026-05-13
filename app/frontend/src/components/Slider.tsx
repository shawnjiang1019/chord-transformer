// src/components/Slider.tsx
// Labeled range input. Track + thumb styles live in globals.css.
// accent-color on the input element handles Firefox/Safari fallback tinting.

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: string;
}

export default function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  accent = '#ff3b00',
}: SliderProps) {
  return (
    <div className="mb-[14px]">
      <div className="flex justify-between font-mono text-[11px] mb-1">
        <span className="text-ink tracking-[0.06em]">{label}</span>
        <span className="font-semibold" style={{ color: accent }}>
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full"
        style={{ accentColor: accent }}
      />
    </div>
  );
}
