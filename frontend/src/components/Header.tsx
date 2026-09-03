import { ExternalLink, Gauge, Loader2, Shapes } from "lucide-react";
import type { Health } from "../types";

const UI_VERSION = "UI v0.3 · React";

interface Props {
  health: Health | null;
  offline: boolean;
}

function HealthBadge({ health, offline }: Props) {
  let tone = "warn";
  let text = "connecting…";
  if (offline) {
    tone = "bad";
    text = "server offline";
  } else if (health) {
    if (health.mock || !health.model_loaded) {
      tone = !health.mock && health.model_loaded ? "bad" : "warn";
      text = health.mock
        ? `mock engine · ${health.device}`
        : `model loading… · ${health.device}`;
    } else {
      tone = "ok";
      text = `model ready · ${health.device}`;
    }
  }
  return (
    <div className={`health health-${tone}`} title={health?.model_id ?? "SAM3"}>
      <span className="health-dot" />
      {!health && !offline ? <Loader2 className="spin-sm" size={12} /> : <Gauge size={12} />}
      <span>{text}</span>
    </div>
  );
}

export default function Header({ health, offline }: Props) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark">
          <Shapes size={20} />
        </div>
        <div>
          <h1>SAM3 Segment Studio</h1>
          <p>Interactive segmentation — text · boxes · points · mixed</p>
        </div>
      </div>
      <div className="header-spacer" />
      <span className="ui-version" title="Frontend build">{UI_VERSION}</span>
      <HealthBadge health={health} offline={offline} />
      <a className="btn btn-ghost btn-sm" href="/docs" target="_blank" rel="noopener noreferrer">
        API docs <ExternalLink size={13} />
      </a>
    </header>
  );
}
