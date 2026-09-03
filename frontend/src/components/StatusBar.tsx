import { AlertCircle, CheckCircle2, Info } from "lucide-react";
import type { StatusMessage } from "../types";

const ICONS = {
  info: Info,
  ok: CheckCircle2,
  err: AlertCircle,
} as const;

export default function StatusBar({ status }: { status: StatusMessage | null }) {
  if (!status) return null;
  const Icon = ICONS[status.kind];
  return (
    <div className={`status status-${status.kind}`} role="status">
      <Icon size={15} className="status-icon" />
      <span dangerouslySetInnerHTML={{ __html: status.text }} />
    </div>
  );
}
