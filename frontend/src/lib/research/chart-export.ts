// Client-side chart export: SVG, PNG and CSV with title, source line and units embedded.
import { MONO } from './palette';

export type ExportMeta = {
	title: string;
	subtitle?: string;
	source?: string; // human-readable source line
	units?: string;
	background?: string;
};

const INLINE_PROPS = [
	'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-opacity', 'opacity',
	'font-family', 'font-size', 'font-weight', 'text-anchor', 'dominant-baseline', 'letter-spacing'
];

function esc(s: string): string {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function inlineStyles(src: Element, dst: Element) {
	const a = [src, ...Array.from(src.querySelectorAll('*'))];
	const b = [dst, ...Array.from(dst.querySelectorAll('*'))];
	for (let i = 0; i < a.length && i < b.length; i++) {
		const cs = getComputedStyle(a[i]);
		const target = b[i] as SVGElement;
		for (const p of INLINE_PROPS) {
			const v = cs.getPropertyValue(p);
			if (v && !target.getAttribute(p)) target.style.setProperty(p, v);
		}
	}
}

export function svgSize(svg: SVGSVGElement): { w: number; h: number } {
	const vb = svg.viewBox?.baseVal;
	if (vb && vb.width && vb.height) return { w: vb.width, h: vb.height };
	const r = svg.getBoundingClientRect();
	return { w: r.width || 640, h: r.height || 300 };
}

/** Serialise a chart <svg> into a standalone SVG string with a title/source/units header and footer. */
export function svgToSvgString(svg: SVGSVGElement, meta: ExportMeta): string {
	const clone = svg.cloneNode(true) as SVGSVGElement;
	inlineStyles(svg, clone);
	clone.querySelectorAll('[data-export-hide]').forEach((n) => n.remove());
	const { w, h } = svgSize(svg);
	const headH = meta.subtitle ? 54 : 36;
	const lines: string[] = [];
	if (meta.source) lines.push(`Source: ${meta.source}`);
	if (meta.units) lines.push(`Units: ${meta.units}`);
	const footH = lines.length ? 14 * lines.length + 12 : 6;
	const total = h + headH + footH;
	const bg = meta.background ?? '#050a15';
	clone.setAttribute('x', '0');
	clone.setAttribute('y', String(headH));
	clone.setAttribute('width', String(w));
	clone.setAttribute('height', String(h));
	clone.removeAttribute('style');
	const head =
		`<text x="14" y="22" font-family="${esc(MONO)}" font-size="14" font-weight="600" fill="#e5f1f0">${esc(meta.title)}</text>` +
		(meta.subtitle ? `<text x="14" y="42" font-family="${esc(MONO)}" font-size="11" fill="#8494a9">${esc(meta.subtitle)}</text>` : '');
	const foot = lines
		.map((l, i) => `<text x="14" y="${headH + h + 16 + i * 14}" font-family="${esc(MONO)}" font-size="10" fill="#64748b">${esc(l)}</text>`)
		.join('');
	const inner = new XMLSerializer().serializeToString(clone);
	return (
		`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${total}" viewBox="0 0 ${w} ${total}">` +
		`<rect width="100%" height="100%" fill="${bg}"/>${head}${inner}${foot}</svg>`
	);
}

/** Rasterise an SVG string to a PNG blob at the given pixel scale. */
export async function svgToPng(svgString: string, scale = 2, background = '#050a15'): Promise<Blob> {
	const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
	const url = URL.createObjectURL(blob);
	try {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('SVG rasterisation failed'));
			img.src = url;
		});
		const w = (img.naturalWidth || img.width) * scale;
		const h = (img.naturalHeight || img.height) * scale;
		const canvas = document.createElement('canvas');
		canvas.width = w;
		canvas.height = h;
		const ctx = canvas.getContext('2d');
		if (!ctx) throw new Error('Canvas unavailable');
		ctx.fillStyle = background;
		ctx.fillRect(0, 0, w, h);
		ctx.drawImage(img, 0, 0, w, h);
		return await new Promise<Blob>((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('PNG encoding failed'))), 'image/png'));
	} finally {
		URL.revokeObjectURL(url);
	}
}

export type CsvColumn = string | { key: string; label?: string };

/** RFC-4180 CSV; non-finite numbers become empty cells. Column labels may carry units. */
export function rowsToCsv(rows: Record<string, unknown>[], columns?: CsvColumn[]): string {
	const cols: { key: string; label: string }[] = (columns ??
		Array.from(rows.reduce((s, r) => { Object.keys(r).forEach((k) => s.add(k)); return s; }, new Set<string>()))
	).map((c) => (typeof c === 'string' ? { key: c, label: c } : { key: c.key, label: c.label ?? c.key }));
	const cell = (v: unknown): string => {
		if (v === null || v === undefined) return '';
		if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
		const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
		return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
	};
	const head = cols.map((c) => cell(c.label)).join(',');
	return [head, ...rows.map((r) => cols.map((c) => cell(r[c.key])).join(','))].join('\r\n') + '\r\n';
}

export function safeFilename(title: string, ext: string): string {
	const base = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'chart';
	return `${base}.${ext}`;
}

export function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}
