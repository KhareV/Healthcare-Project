// Small scale / layout helpers shared by the SVG charts.
export function extent(values: number[]): [number, number] | null {
	let lo = Infinity;
	let hi = -Infinity;
	for (const v of values) {
		if (Number.isFinite(v)) {
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
	}
	return lo <= hi ? [lo, hi] : null;
}

export function niceTicks(min: number, max: number, count = 5): number[] {
	if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
	if (min === max) return [min];
	const span = max - min;
	const raw = span / Math.max(1, count);
	const mag = Math.pow(10, Math.floor(Math.log10(raw)));
	const norm = raw / mag;
	const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
	const start = Math.ceil(min / step - 1e-9) * step;
	const out: number[] = [];
	for (let v = start; v <= max + step * 1e-6; v += step) out.push(Number(v.toFixed(10)));
	return out;
}

export function pad(domain: [number, number], frac = 0.05): [number, number] {
	const span = domain[1] - domain[0] || Math.abs(domain[0]) || 1;
	return [domain[0] - span * frac, domain[1] + span * frac];
}

export function linear(d0: number, d1: number, r0: number, r1: number) {
	const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
	return (v: number) => r0 + (v - d0) * k;
}

export function approxTextWidth(text: string, size = 10): number {
	return text.length * size * 0.62;
}

export type LegendItem = { key: string; label: string; color: string; dash?: string; hidden?: boolean };
export type LegendLayout = { items: (LegendItem & { x: number; y: number })[]; height: number };

export function layoutLegend(items: LegendItem[], width: number, rowHeight = 16): LegendLayout {
	const out: (LegendItem & { x: number; y: number })[] = [];
	let x = 0;
	let row = 0;
	for (const it of items) {
		const w = 26 + approxTextWidth(it.label, 10) + 14;
		if (x > 0 && x + w > width) {
			x = 0;
			row += 1;
		}
		out.push({ ...it, x, y: row * rowHeight });
		x += w;
	}
	return { items: out, height: items.length ? (row + 1) * rowHeight : 0 };
}

// Deterministic jitter in [-1, 1] from an integer seed (no Math.random: exports must be reproducible).
export function jitter(seed: number): number {
	const s = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
	return (s - Math.floor(s)) * 2 - 1;
}
