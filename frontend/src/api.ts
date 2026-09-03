import type { Health, SegmentationResult, SegmentSettings } from "./types";

/**
 * API base URL for the backend.
 *
 * - Empty by default → same-origin calls (Vite dev/preview proxy, or Nginx
 *   reverse proxy in Docker) — nothing to configure.
 * - Set VITE_API_BASE at build time to point at a standalone backend, e.g.
 *   `VITE_API_BASE=http://localhost:8000 npm run build`.
 */
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined ?? "").replace(/\/$/, "");

/** Build the multipart payload for POST /segment. */
export function buildSegmentForm(
  file: File,
  fileName: string,
  mode: string,
  text: string,
  pointsPositive: number[][],
  pointsNegative: number[][],
  boxesPositive: number[][],
  boxesNegative: number[][],
  settings: SegmentSettings,
): FormData {
  const fd = new FormData();
  fd.append("image", file, fileName);
  fd.append("mode", mode);
  fd.append("text", text);
  fd.append("points_positive", JSON.stringify(pointsPositive));
  fd.append("points_negative", JSON.stringify(pointsNegative));
  fd.append("boxes_positive", JSON.stringify(boxesPositive));
  fd.append("boxes_negative", JSON.stringify(boxesNegative));
  fd.append("score_threshold", String(settings.score));
  fd.append("mask_threshold", String(settings.mask));
  fd.append("opacity", String(settings.opacity));
  fd.append("max_masks", String(settings.maxMasks));
  fd.append("show_semantic", String(settings.showSemantic));
  return fd;
}

export async function fetchHealth(): Promise<Health> {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json() as Promise<Health>;
}

export async function runSegmentation(
  file: File,
  fileName: string,
  mode: string,
  text: string,
  pointsPositive: number[][],
  pointsNegative: number[][],
  boxesPositive: number[][],
  boxesNegative: number[][],
  settings: SegmentSettings,
): Promise<SegmentationResult> {
  const fd = buildSegmentForm(
    file,
    fileName,
    mode,
    text,
    pointsPositive,
    pointsNegative,
    boxesPositive,
    boxesNegative,
    settings,
  );
  const resp = await fetch(`${API_BASE}/segment`, { method: "POST", body: fd });
  const data = (await resp.json().catch(() => ({}))) as Partial<SegmentationResult> & {
    detail?: string;
  };
  if (!resp.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${resp.status}`);
  }
  return data as SegmentationResult;
}
