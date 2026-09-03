/** Shared frontend types. */

export type Point = [number, number];
export type Box = [number, number, number, number];
export type Tool = "pp" | "pn" | "bp" | "bn";
export type Mode = "auto" | "text" | "box" | "point" | "mixed";

export type PromptKind = "pp" | "pn" | "bp" | "bn";

export interface HistoryEntry {
  kind: PromptKind;
  index: number;
}

export interface PromptState {
  pp: Point[];
  pn: Point[];
  bp: Box[];
  bn: Box[];
  history: HistoryEntry[];
}

export interface LoadedImage {
  file: File;
  url: string;
  bitmap: HTMLImageElement;
  name: string;
  width: number;
  height: number;
  sizeBytes: number;
}

export interface Health {
  status: string;
  mock: boolean;
  device: string;
  model_loaded: boolean;
  lazy_load: boolean;
  model_id: string;
  hf_auth: string;
}

export interface MaskInstance {
  id: number;
  source: string;
  score: number | null;
  box: number[];
  area_px: number;
  mask: string;
}

export interface SegmentationResult {
  status: string;
  run_id: string | null;
  mode: string;
  prompt: string;
  elapsed_seconds: number;
  num_instances: number;
  composite: string;
  semantic: string | null;
  warnings: string[];
  instances: MaskInstance[];
  files: {
    composite?: string;
    json?: string;
    masks?: string[];
  };
  image_size?: [number, number];
}

export interface SegmentSettings {
  score: number;
  mask: number;
  opacity: number;
  maxMasks: number;
  showSemantic: boolean;
}

export interface StatusMessage {
  kind: "info" | "ok" | "err";
  text: string;
}
