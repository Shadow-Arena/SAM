import { Crosshair, MousePointer2, Play, Square, SquareDashed, Trash2, Undo2 } from "lucide-react";
import type { Mode, PromptState, Tool } from "../types";
import { MODE_OPTIONS, TOOL_META, promptCount } from "../lib/editor";
import AdvancedSettings, { type Settings } from "./AdvancedSettings";

interface Props {
  tool: Tool;
  prompts: PromptState;
  mode: Mode;
  text: string;
  settings: Settings;
  running: boolean;
  onTool(tool: Tool): void;
  onMode(mode: Mode): void;
  onText(text: string): void;
  onSettings(next: Settings): void;
  onRun(): void;
  onUndo(): void;
  onClear(): void;
}

const TOOL_ICONS: Record<Tool, typeof Crosshair> = {
  pp: Crosshair,
  pn: MousePointer2,
  bp: Square,
  bn: SquareDashed,
};

function Chips({ prompts }: { prompts: PromptState }) {
  const chips: { key: keyof PromptState; label: string; tone: string }[] = [
    { key: "pp", label: "points +", tone: "pos" },
    { key: "pn", label: "points −", tone: "neg" },
    { key: "bp", label: "boxes +", tone: "pos" },
    { key: "bn", label: "boxes −", tone: "neg" },
  ];
  const visible = chips.filter((c) => prompts[c.key].length > 0);
  if (!visible.length) return <div className="chips-empty">No prompts yet — click or drag on the image.</div>;
  return (
    <div className="chips">
      {visible.map((c) => (
        <span key={c.key} className="chip">
          <span className={`chip-dot chip-${c.tone}`} />
          {prompts[c.key].length} {c.label}
        </span>
      ))}
    </div>
  );
}

export default function PromptPanel({
  tool,
  prompts,
  mode,
  text,
  settings,
  running,
  onTool,
  onMode,
  onText,
  onSettings,
  onRun,
  onUndo,
  onClear,
}: Props) {
  const count = promptCount(prompts);
  return (
    <section className="card">
      <div className="card-head">
        <Crosshair size={15} className="card-icon" />
        <h2>Prompts</h2>
        <span className="spacer" />
        <span className={`badge ${count ? "badge-active" : ""}`}>{count}</span>
      </div>
      <div className="card-body">
        <div className="tool-grid" role="toolbar" aria-label="Prompt tools">
          {(Object.keys(TOOL_META) as Tool[]).map((t) => {
            const Icon = TOOL_ICONS[t];
            const meta = TOOL_META[t];
            return (
              <button
                key={t}
                className={`tool tool-${meta.tone} ${tool === t ? "active" : ""}`}
                onClick={() => onTool(t)}
                title={meta.short}
                aria-pressed={tool === t}
              >
                <ImIcon icon={Icon} tone={meta.tone} />
                <span className="tool-label">{meta.label}</span>
                <span className="kbd">{meta.kbd}</span>
              </button>
            );
          })}
        </div>

        <Chips prompts={prompts} />

        <div className="field">
          <label htmlFor="text">Text prompt <span className="field-optional">(optional)</span></label>
          <input
            id="text"
            type="text"
            value={text}
            placeholder='e.g. "yellow school bus", "ear", "handle"'
            onChange={(e) => onText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) onRun();
            }}
          />
        </div>

        <div className="field">
          <label htmlFor="mode">Mode</label>
          <select id="mode" value={mode} onChange={(e) => onMode(e.target.value as Mode)}>
            {MODE_OPTIONS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label} — {m.desc}
              </option>
            ))}
          </select>
        </div>

        <AdvancedSettings settings={settings} onChange={onSettings} />

        <button className="btn btn-primary btn-wide btn-run" onClick={onRun} disabled={running}>
          {running ? (
            <>
              <span className="spinner" /> Running…
            </>
          ) : (
            <>
              <Play size={15} /> Run segmentation
            </>
          )}
        </button>
        <div className="split-buttons">
          <button className="btn" onClick={onUndo} disabled={!prompts.history.length}>
            <Undo2 size={14} /> Undo
          </button>
          <button className="btn" onClick={onClear} disabled={!count}>
            <Trash2 size={14} /> Clear
          </button>
        </div>
      </div>
    </section>
  );
}

function ImIcon({ icon: Icon, tone }: { icon: typeof Crosshair; tone: string }) {
  return (
    <span className={`tool-icon tool-icon-${tone}`}>
      <Icon size={13} />
    </span>
  );
}
