// A small dependency-free SVG line chart for per-epoch training metrics.
export interface Series {
  label: string;
  color: string;
  values: number[];
  dashed?: boolean;
}

const W = 320;
const H = 150;
const PAD = { left: 38, right: 8, top: 10, bottom: 22 };

export function MetricChart({ title, series, epochs, yMax, format = (v) => v.toFixed(2), testId }: {
  title: string;
  series: Series[];
  epochs: number;
  yMax?: number;
  format?: (value: number) => string;
  testId?: string;
}) {
  const values = series.flatMap((s) => s.values).filter(Number.isFinite);
  const top = yMax ?? Math.max(1e-6, ...values) * 1.05;
  const xSpan = Math.max(epochs - 1, 1);
  const x = (i: number) => PAD.left + (i / xSpan) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - Math.min(v, top) / top) * (H - PAD.top - PAD.bottom);
  const latest = series.map((s) => `${s.label} ${s.values.length ? format(s.values[s.values.length - 1]) : "—"}`).join(", ");
  return (
    <figure className="metric-chart" data-testid={testId}>
      <figcaption>
        <span>{title}</span>
        <span className="metric-legend">
          {series.map((s) => (
            <span key={s.label}><i style={{ background: s.color }} className={s.dashed ? "dashed" : ""} />{s.label}</span>
          ))}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${latest}`}>
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(top * t)} y2={y(top * t)} className="grid" />
            <text x={PAD.left - 4} y={y(top * t) + 3} textAnchor="end">{format(top * t)}</text>
          </g>
        ))}
        <text x={PAD.left} y={H - 6}>1</text>
        <text x={W - PAD.right} y={H - 6} textAnchor="end">{epochs} epochs</text>
        {series.map((s) => s.values.length > 0 && (
          <polyline key={s.label} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined}
            points={s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")} />
        ))}
      </svg>
    </figure>
  );
}
