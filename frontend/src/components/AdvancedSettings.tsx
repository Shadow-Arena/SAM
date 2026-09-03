import { ChevronDown } from "lucide-react";

export interface Settings {
  score: number;
  mask: number;
  opacity: number;
  maxMasks: number;
  showSemantic: boolean;
}

interface Props {
  settings: Settings;
  onChange(next: Settings): void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  digits,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits: number;
  onChange(value: number): void;
}) {
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{label}</span>
        <span className="slider-val">{value.toFixed(digits)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function AdvancedSettings({ settings, onChange }: Props) {
  const set = (patch: Partial<Settings>) => onChange({ ...settings, ...patch });
  return (
    <details className="adv">
      <summary>
        <ChevronDown size={14} /> Advanced settings
      </summary>
      <div className="adv-body">
        <Slider
          label="Score threshold"
          value={settings.score}
          min={0}
          max={1}
          step={0.01}
          digits={2}
          onChange={(v) => set({ score: v })}
        />
        <Slider
          label="Mask threshold"
          value={settings.mask}
          min={0}
          max={1}
          step={0.01}
          digits={2}
          onChange={(v) => set({ mask: v })}
        />
        <Slider
          label="Overlay opacity"
          value={settings.opacity}
          min={0}
          max={1}
          step={0.05}
          digits={2}
          onChange={(v) => set({ opacity: v })}
        />
        <Slider
          label="Max masks"
          value={settings.maxMasks}
          min={1}
          max={100}
          step={1}
          digits={0}
          onChange={(v) => set({ maxMasks: v })}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={settings.showSemantic}
            onChange={(e) => set({ showSemantic: e.target.checked })}
          />
          Show semantic mask (text mode)
        </label>
      </div>
    </details>
  );
}
