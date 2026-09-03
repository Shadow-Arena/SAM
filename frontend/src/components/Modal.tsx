import { X } from "lucide-react";

interface Props {
  src: string | null;
  onClose(): void;
}

export default function Modal({ src, onClose }: Props) {
  if (!src) return null;
  return (
    <div className="modal" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <button className="modal-close" onClick={onClose} aria-label="Close">
          <X size={15} />
        </button>
        <img src={src} alt="preview" />
      </div>
    </div>
  );
}
