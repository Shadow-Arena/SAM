import { useRef } from "react";
import { Image as ImageIcon, RefreshCw, Upload } from "lucide-react";
import type { LoadedImage } from "../types";
import { formatBytes } from "../lib/editor";

interface Props {
  image: LoadedImage | null;
  onFile(file: File): void;
}

export default function ImagePanel({ image, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const openPicker = () => inputRef.current?.click();

  return (
    <section className="card">
      <div className="card-head">
        <ImageIcon size={15} className="card-icon" />
        <h2>Image</h2>
      </div>
      <div className="card-body">
        {image ? (
          <div className="image-row">
            <img className="thumb" src={image.url} alt="" />
            <div className="thumb-meta">
              <div className="meta-name" title={image.name}>
                {image.name}
              </div>
              <div className="meta-dim">
                {image.width} × {image.height} px · {formatBytes(image.sizeBytes)}
              </div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={openPicker} title="Choose another image">
              <RefreshCw size={13} /> Change
            </button>
          </div>
        ) : (
          <button
            className="dropzone"
            onClick={openPicker}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer?.files?.[0];
              if (file) onFile(file);
            }}
          >
            <span className="dz-icon">
              <Upload size={20} />
            </span>
            <span className="dz-title">Drop an image here</span>
            <span className="dz-sub">PNG · JPEG · WebP — or click to browse</span>
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
        <p className="hint">
          You can also paste an image (<kbd>Ctrl</kbd>+<kbd>V</kbd>) or drop it anywhere.
        </p>
      </div>
    </section>
  );
}
