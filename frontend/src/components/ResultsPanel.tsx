import { useState } from "react";
import { AlertTriangle, BarChart3, Download, Layers, Table2 } from "lucide-react";
import type { LoadedImage, SegmentationResult } from "../types";

interface Props {
  result: SegmentationResult | null;
  original: LoadedImage | null;
  onZoom(src: string): void;
}

export default function ResultsPanel({ result, original, onZoom }: Props) {
  const [view, setView] = useState<"result" | "original">("result");

  if (!result) {
    return (
      <section className="card">
        <div className="card-head">
          <BarChart3 size={15} className="card-icon" />
          <h2>Results</h2>
        </div>
        <div className="card-body">
          <div className="results-empty">
            <div className="empty-icon">📊</div>
            <p>Results will appear here after you run a segmentation.</p>
          </div>
        </div>
      </section>
    );
  }

  const maskCount = result.files?.masks?.length ?? 0;
  const entries: Array<[string, string | undefined]> = [
    ["Composite", result.files?.composite],
    ["Result JSON", result.files?.json],
  ];
  if (maskCount > 0) entries.push([`Masks (${maskCount})`, result.files?.masks?.[0]]);
  const downloads = entries.filter(([, url]) => Boolean(url));

  return (
    <section className="card">
      <div className="card-head">
        <BarChart3 size={15} className="card-icon" />
        <h2>Results</h2>
        <span className="spacer" />
        <div className="compare" role="tablist" aria-label="Result view">
          <button className={view === "result" ? "active" : ""} onClick={() => setView("result")} role="tab">
            Segmented
          </button>
          <button
            className={view === "original" ? "active" : ""}
            onClick={() => setView("original")}
            role="tab"
            disabled={!original}
          >
            Original
          </button>
        </div>
      </div>
      <div className="card-body">
        <div className="result-meta">
          <span className="stat">
            <b>{result.num_instances}</b> objects
          </span>
          <span className="stat">
            <b>{result.elapsed_seconds}s</b> elapsed
          </span>
          <span className="stat">
            mode <b>{result.mode}</b>
          </span>
          <span className="stat">
            prompts <b>{result.prompt}</b>
          </span>
        </div>

        {result.warnings.length > 0 && (
          <div className="warnings">
            <AlertTriangle size={14} />
            {result.warnings.join(" · ")}
          </div>
        )}

        <img
          className="result-img"
          src={view === "result" ? result.composite : original?.url}
          alt="segmentation result"
          onClick={() => onZoom(view === "result" ? result.composite : original?.url ?? "")}
          title="Click to zoom"
        />

        {result.instances.length > 0 && (
          <>
            <div className="divider" />
            <div className="sec-title">
              <Table2 size={14} /> Detected objects
            </div>
            <div className="table-wrap">
              <table className="objects">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Source</th>
                    <th>Score</th>
                    <th>Box (x1, y1, x2, y2)</th>
                    <th>Area px</th>
                  </tr>
                </thead>
                <tbody>
                  {result.instances.map((inst) => (
                    <tr key={inst.id}>
                      <td>
                        <span className="obj-id">{inst.id}</span>
                      </td>
                      <td className="src">{inst.source}</td>
                      <td className={inst.score == null ? "" : "score-pill"}>
                        {inst.score == null ? "—" : inst.score.toFixed(3)}
                      </td>
                      <td className="mono">{inst.box.join(", ")}</td>
                      <td className="mono">{inst.area_px.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {result.instances.length > 0 && (
          <>
            <div className="divider" />
            <div className="sec-title">
              <Layers size={14} /> Masks
            </div>
            <div className="masks">
              {result.instances.map((inst) => (
                <button key={inst.id} className="mask-item" onClick={() => onZoom(inst.mask)} title="Click to zoom">
                  <img src={inst.mask} alt={`mask ${inst.id}`} loading="lazy" />
                  <span className="mask-name">
                    #{inst.id} · {inst.source}
                    <b>{inst.score == null ? "—" : inst.score.toFixed(2)}</b>
                  </span>
                </button>
              ))}
            </div>
          </>
        )}

        {downloads.length > 0 && (
          <>
            <div className="divider" />
            <div className="sec-title">
              <Download size={14} /> Downloads
            </div>
            <div className="downloads">
              {downloads.map(([label, url]) => (
                <a key={label} className="btn btn-sm" href={url} download>
                  <Download size={13} /> {label}
                </a>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
