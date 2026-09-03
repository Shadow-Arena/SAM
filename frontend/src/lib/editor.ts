import type { Box, Point, PromptState, Tool } from "../types";

export const TOOL_META: Record<Tool, { label: string; short: string; kbd: string; tone: "pos" | "neg" }> = {
  pp: { label: "Point +", short: "Positive point — target this pixel", kbd: "1", tone: "pos" },
  pn: { label: "Point −", short: "Negative point — avoid this pixel", kbd: "2", tone: "neg" },
  bp: { label: "Box +", short: "Positive box — region to segment", kbd: "3", tone: "pos" },
  bn: { label: "Box −", short: "Negative box — region to exclude", kbd: "4", tone: "neg" },
};

export const MODE_OPTIONS = [
  { value: "auto", label: "Auto", desc: "text/boxes → PCS · points → tracker" },
  { value: "text", label: "Text", desc: "text (+ optional boxes)" },
  { value: "box", label: "Box", desc: "boxes only" },
  { value: "point", label: "Point", desc: "points only" },
  { value: "mixed", label: "Mixed", desc: "text/boxes + points" },
] as const;

export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 8;
export const DRAG_THRESHOLD_IMAGE_PX = 4;

export function clampZoom(v: number): number {
  return Math.min(Math.max(v, MIN_ZOOM), MAX_ZOOM);
}

/** Convert a pointer event position to image pixel coordinates. */
export function toImageXY(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, "left" | "top">,
  scale: number,
  imageWidth: number,
  imageHeight: number,
): Point {
  const x = (clientX - rect.left) / scale;
  const y = (clientY - rect.top) / scale;
  return [
    Math.min(Math.max(Math.round(x), 0), imageWidth - 1),
    Math.min(Math.max(Math.round(y), 0), imageHeight - 1),
  ];
}

export function normalizeBox(a: Point, b: Point): Box {
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[0], b[0]),
    Math.max(a[1], b[1]),
  ];
}

export function manhattan(a: Point, b: Point): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

export function promptCount(p: PromptState): number {
  return p.pp.length + p.pn.length + p.bp.length + p.bn.length;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
