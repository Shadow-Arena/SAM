import { useCallback, useEffect, useRef, useState } from "react";
import { ImagePlus, Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import type { LoadedImage, Point, PromptState, Tool } from "../types";
import {
  DRAG_THRESHOLD_IMAGE_PX,
  MAX_ZOOM,
  MIN_ZOOM,
  clampZoom,
  manhattan,
  normalizeBox,
  toImageXY,
} from "../lib/editor";

interface DragState {
  start: Point;
  cur: Point;
  negative: boolean;
  moved: boolean;
}

interface Props {
  image: LoadedImage | null;
  tool: Tool;
  prompts: PromptState;
  view: number;
  onViewChange(view: number): void;
  onAddPrompt(kind: "pp" | "pn" | "bp" | "bn", value: Point | number[]): void;
}

const POS = "#34d399";
const NEG = "#f87171";
const POS_FILL = "rgba(52, 211, 153, .16)";
const NEG_FILL = "rgba(248, 113, 113, .16)";

export default function EditorCanvas({ image, tool, prompts, view, onViewChange, onAddPrompt }: Props) {
  const shellRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fit, setFit] = useState(1);
  const [drag, setDrag] = useState<DragState | null>(null);

  // Refs mirror the latest values so the window-level pointer listeners
  // never read stale state (drag must keep working outside the canvas).
  const imageRef = useRef(image);
  const toolRef = useRef(tool);
  const viewRef = useRef(view);
  const scaleRef = useRef(1);
  const onViewChangeRef = useRef(onViewChange);
  const onAddPromptRef = useRef(onAddPrompt);
  const dragRef = useRef<DragState | null>(null);

  imageRef.current = image;
  toolRef.current = tool;
  viewRef.current = view;
  onViewChangeRef.current = onViewChange;
  onAddPromptRef.current = onAddPrompt;

  const scale = fit * view;
  scaleRef.current = scale;

  /* ------------------------------------------------------------ fit / layout */

  const computeFit = useCallback(() => {
    const shell = shellRef.current;
    const img = imageRef.current;
    if (!shell || !img) return;
    const availW = Math.max(shell.clientWidth - 28, 80);
    const availH = Math.max(shell.clientHeight - 28, 200);
    const f = Math.min(availW / img.width, availH / img.height, 1);
    setFit((prev) => (Math.abs(prev - f) > 1e-4 ? f : prev));
  }, []);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const ro = new ResizeObserver(() => computeFit());
    ro.observe(shell);
    return () => ro.disconnect();
  }, [computeFit]);

  useEffect(() => {
    if (image) {
      setFit(1);
      requestAnimationFrame(computeFit);
    }
  }, [image, computeFit]);

  /* ------------------------------------------------------------ drawing */

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !image) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(image.width * scale));
    const h = Math.max(1, Math.round(image.height * scale));
    const canvas = canvasRef.current!;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(image.bitmap, 0, 0, w, h);
    ctx.lineJoin = "round";

    const drawBox = (b: number[], color: string, fill: string) => {
      const x = b[0] * scale;
      const y = b[1] * scale;
      const bw = (b[2] - b[0]) * scale;
      const bh = (b[3] - b[1]) * scale;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, bw, bh);
      ctx.setLineDash([]);
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, bw, bh);
    };

    prompts.bp.forEach((b) => drawBox(b, POS, POS_FILL));
    prompts.bn.forEach((b) => drawBox(b, NEG, NEG_FILL));

    const drawPoint = (p: Point, color: string, sign: string) => {
      const x = p[0] * scale;
      const y = p[1] * scale;
      const r = 9;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "rgba(255,255,255,.95)";
      ctx.stroke();
      ctx.fillStyle = "#0a0f18";
      ctx.font = `700 ${Math.round(r * 1.05)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(sign, x, y + 0.5);
    };

    prompts.pp.forEach((p) => drawPoint(p, POS, "+"));
    prompts.pn.forEach((p) => drawPoint(p, NEG, "−"));

    if (drag) {
      const box = normalizeBox(drag.start, drag.cur);
      drawBox(box, drag.negative ? NEG : POS, drag.negative ? NEG_FILL : POS_FILL);
    }
  }, [image, prompts, drag, scale]);

  /* ------------------------------------------------------------ pointer */

  const posFromEvent = useCallback((e: PointerEvent | React.PointerEvent): Point => {
    const canvas = canvasRef.current!;
    return toImageXY(e.clientX, e.clientY, canvas.getBoundingClientRect(), scaleRef.current, imageRef.current!.width, imageRef.current!.height);
  }, []);

  // move/up on window so a drag keeps working when the pointer leaves the
  // canvas (the old canvas-only listeners created the stray-corner point).
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const d = dragRef.current;
      if (!d || !imageRef.current) return;
      const cur = posFromEvent(e);
      const next = {
        ...d,
        cur,
        moved: d.moved || manhattan(d.start, cur) > DRAG_THRESHOLD_IMAGE_PX,
      };
      dragRef.current = next;
      setDrag(next);
    };
    const onUp = (e: PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      dragRef.current = null;
      setDrag(null);
      const end = posFromEvent(e);
      if (d.moved) {
        onAddPromptRef.current(d.negative ? "bn" : "bp", normalizeBox(d.start, end));
      } else {
        onAddPromptRef.current(d.negative ? "pn" : "pp", end);
      }
    };
    const onWindowUp = (e: PointerEvent) => onUp(e);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onWindowUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onWindowUp);
    };
  }, [posFromEvent]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!image) return;
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* capture is best-effort */
    }
    const pos = posFromEvent(e);
    const t = toolRef.current;
    if (t === "bp" || t === "bn") {
      const next: DragState = { start: pos, cur: pos, negative: t === "bn", moved: false };
      dragRef.current = next;
      setDrag(next);
    } else {
      const kind = t === "pn" || e.shiftKey ? "pn" : "pp";
      onAddPrompt(kind, pos);
    }
  };

  /* ------------------------------------------------------------ zoom */

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const onWheel = (e: WheelEvent) => {
      if (!imageRef.current) return;
      e.preventDefault();
      onViewChangeRef.current(
        clampZoom(viewRef.current * (e.deltaY < 0 ? 1.15 : 1 / 1.15)),
      );
    };
    shell.addEventListener("wheel", onWheel, { passive: false });
    return () => shell.removeEventListener("wheel", onWheel);
  }, []);

  const zoomIn = () => onViewChange(clampZoom(view * 1.25));
  const zoomOut = () => onViewChange(clampZoom(view / 1.25));
  const zoomFit = () => onViewChange(1);

  const pct = Math.round(scale * 100);
  const canZoom = view < MAX_ZOOM;

  return (
    <section className="card editor">
      <div className="editor-toolbar">
        <div className="editor-info">
          <div className="editor-name">{image?.name ?? "No image loaded"}</div>
          <div className="editor-dims">
            {image
              ? `${image.width} × ${image.height} px · scroll or wheel to inspect`
              : "upload or drop an image on the left"}
          </div>
        </div>
        <div className="zoom-group" role="group" aria-label="Zoom">
          <button onClick={zoomOut} disabled={view <= MIN_ZOOM} title="Zoom out (−)">
            <ZoomOut size={14} />
          </button>
          <button className="zoom-fit" onClick={zoomFit} title="Fit to screen (0)">
            {pct}%
          </button>
          <button onClick={zoomIn} disabled={!canZoom} title="Zoom in (+)">
            <ZoomIn size={14} />
          </button>
          <span className="zoom-sep" />
          <button onClick={zoomFit} title="Fit to screen">
            <Maximize2 size={13} />
          </button>
        </div>
      </div>

      <div ref={shellRef} className="canvas-shell">
        <div className="canvas-anchor">
          <canvas
            ref={canvasRef}
            onPointerDown={onPointerDown}
            onContextMenu={(e) => e.preventDefault()}
            aria-label="Segmentation prompt canvas"
          />
        </div>
        {!image && (
          <div className="empty-state">
            <div className="empty-icon">
              <ImagePlus size={34} strokeWidth={1.6} />
            </div>
            <p className="empty-title">No image loaded</p>
            <p className="empty-sub">Upload one in the sidebar, or drag &amp; paste it anywhere.</p>
          </div>
        )}
      </div>

      <div className="stage-foot">
        <span>
          <b>Click</b> = point · <b>drag</b> = box · <b>Shift</b>+click = opposite sign
        </span>
        <span className="spacer" />
        <span>
          <kbd>1–4</kbd> tool&nbsp; <kbd>⌫</kbd> undo&nbsp; <kbd>Ctrl</kbd>+<kbd>Enter</kbd> run
        </span>
      </div>
    </section>
  );
}
