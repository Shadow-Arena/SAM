import { useCallback, useEffect, useRef, useState } from "react";
import { fetchHealth, runSegmentation } from "./api";
import EditorCanvas from "./components/EditorCanvas";
import Header from "./components/Header";
import ImagePanel from "./components/ImagePanel";
import Modal from "./components/Modal";
import PromptPanel from "./components/PromptPanel";
import ResultsPanel from "./components/ResultsPanel";
import StatusBar from "./components/StatusBar";
import { clampZoom, promptCount } from "./lib/editor";
import type {
  Box,
  Health,
  LoadedImage,
  Mode,
  Point,
  PromptKind,
  PromptState,
  SegmentationResult,
  StatusMessage,
  Tool,
} from "./types";

const EMPTY_PROMPTS: PromptState = { pp: [], pn: [], bp: [], bn: [], history: [] };

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [tool, setTool] = useState<Tool>("pp");
  const [prompts, setPrompts] = useState<PromptState>(EMPTY_PROMPTS);
  const [view, setView] = useState(1);
  const [mode, setMode] = useState<Mode>("auto");
  const [text, setText] = useState("");
  const [settings, setSettings] = useState({
    score: 0.3,
    mask: 0.5,
    opacity: 0.55,
    maxMasks: 100,
    showSemantic: false,
  });
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [modalSrc, setModalSrc] = useState<string | null>(null);

  const loadFileRef = useRef<(file: File) => void>(() => {});
  const runRef = useRef<() => void>(() => {});

  /* ------------------------------------------------------------ helpers */

  const addPrompt = useCallback((kind: PromptKind, value: Point | Box) => {
    setPrompts((s) => ({
      ...s,
      [kind]: [...s[kind], value],
      history: [...s.history, { kind, index: s[kind].length }],
    }));
  }, []);

  const undo = useCallback(() => {
    setPrompts((s) => {
      const last = s.history[s.history.length - 1];
      if (!last) return s;
      const list = s[last.kind];
      return {
        ...s,
        [last.kind]: list.filter((_, i) => i !== last.index),
        history: s.history.slice(0, -1),
      };
    });
  }, []);

  const clear = useCallback(() => {
    setPrompts(EMPTY_PROMPTS);
    setResult(null);
  }, []);

  const loadFile = useCallback((file: File) => {
    if (!/^image\/(png|jpe?g|webp|bmp|gif)$/i.test(file.type)) {
      setStatus({ kind: "err", text: "Unsupported file type. Use PNG, JPEG or WebP." });
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return {
          file,
          url,
          bitmap: img,
          name: file.name || "pasted-image.png",
          width: img.naturalWidth,
          height: img.naturalHeight,
          sizeBytes: file.size,
        };
      });
      setPrompts(EMPTY_PROMPTS);
      setResult(null);
      setView(1);
      setStatus({
        kind: "ok",
        text: `Image loaded (${img.naturalWidth}×${img.naturalHeight}). Click or drag on the image to add prompts.`,
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus({ kind: "err", text: "Could not decode that image." });
    };
    img.src = url;
  }, []);
  loadFileRef.current = loadFile;

  /* ------------------------------------------------------------ run */

  const run = useCallback(async () => {
    if (!image) {
      setStatus({ kind: "err", text: "Upload an image first." });
      return;
    }
    const count = promptCount(prompts);
    if (!text.trim() && count === 0) {
      setStatus({ kind: "err", text: "Draw points/boxes on the image or type a text prompt." });
      return;
    }
    setRunning(true);
    setStatus({ kind: "info", text: "Running segmentation… this can take a moment on CPU." });
    try {
      const data = await runSegmentation(
        image.file,
        image.name,
        mode,
        text.trim(),
        prompts.pp,
        prompts.pn,
        prompts.bp,
        prompts.bn,
        settings,
      );
      setResult(data);
      setStatus({
        kind: "ok",
        text: `<b>${data.num_instances}</b> object(s) in <b>${data.elapsed_seconds}s</b> — mode <b>${data.mode}</b> · ${data.prompt}`,
      });
    } catch (err) {
      setStatus({ kind: "err", text: String(err instanceof Error ? err.message : err) });
    } finally {
      setRunning(false);
    }
  }, [image, mode, text, prompts, settings]);
  runRef.current = run;

  /* ------------------------------------------------------------ effects */

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await fetchHealth();
        if (alive) {
          setHealth(h);
          setOffline(false);
        }
      } catch {
        if (alive) {
          setOffline(true);
          setHealth(null);
        }
      }
    };
    poll();
    const id = window.setInterval(poll, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (file) loadFileRef.current(file);
    };
    const onPaste = (e: ClipboardEvent) => {
      for (const item of e.clipboardData?.items ?? []) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) loadFileRef.current(file);
          return;
        }
      }
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("paste", onPaste);
    };
  }, []);

  useEffect(() => {
    const keymap: Record<string, Tool> = { "1": "pp", "2": "pn", "3": "bp", "4": "bn" };
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (keymap[e.key]) {
        setTool(keymap[e.key]);
        e.preventDefault();
      } else if (e.key === "Backspace" || (e.key.toLowerCase() === "z" && (e.ctrlKey || e.metaKey))) {
        undo();
        e.preventDefault();
      } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        runRef.current();
        e.preventDefault();
      } else if (e.key === "Escape") {
        setModalSrc(null);
      } else if (e.key === "+" || e.key === "=") {
        setView((v) => clampZoom(v * 1.25));
      } else if (e.key === "-") {
        setView((v) => clampZoom(v / 1.25));
      } else if (e.key === "0") {
        setView(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo]);

  /* ------------------------------------------------------------ render */

  return (
    <div className="shell">
      <Header health={health} offline={offline} />
      <div className="layout">
        <aside className="sidebar">
          <ImagePanel image={image} onFile={loadFile} />
          <PromptPanel
            tool={tool}
            prompts={prompts}
            mode={mode}
            text={text}
            settings={settings}
            running={running}
            onTool={setTool}
            onMode={setMode}
            onText={setText}
            onSettings={setSettings}
            onRun={run}
            onUndo={undo}
            onClear={clear}
          />
        </aside>
        <main className="main">
          <EditorCanvas
            image={image}
            tool={tool}
            prompts={prompts}
            view={view}
            onViewChange={setView}
            onAddPrompt={addPrompt}
          />
          <StatusBar status={status} />
          <ResultsPanel result={result} original={image} onZoom={setModalSrc} />
        </main>
      </div>
      <Modal src={modalSrc} onClose={() => setModalSrc(null)} />
    </div>
  );
}
