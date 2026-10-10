// Unit-aware number formatting. Never conflates SOFA units, hours and probabilities.
export function isNum(v: unknown): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}
export function fmtNum(v: number | null | undefined, digits = 3): string {
	return isNum(v) ? v.toFixed(digits) : '—';
}
export function fmtInt(v: number | null | undefined): string {
	return isNum(v) ? Math.round(v).toLocaleString('en-US') : '—';
}
export function fmtProb(v: number | null | undefined, digits = 3): string {
	return isNum(v) ? v.toFixed(digits) : '—';
}
export function fmtPct(v: number | null | undefined, digits = 1): string {
	return isNum(v) ? `${(v * 100).toFixed(digits)}%` : '—';
}
export function fmtHours(v: number | null | undefined, digits = 1): string {
	return isNum(v) ? `${v.toFixed(digits)} h` : '—';
}
export function fmtSofa(v: number | null | undefined, digits = 2): string {
	return isNum(v) ? `${v.toFixed(digits)} SOFA` : '—';
}
export function fmtSigned(v: number | null | undefined, digits = 3): string {
	return isNum(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(digits)}` : '—';
}
export function fmtCI(lo: number | null | undefined, hi: number | null | undefined, digits = 3): string {
	return isNum(lo) && isNum(hi) ? `[${lo.toFixed(digits)}, ${hi.toFixed(digits)}]` : '—';
}
export function withUnit(text: string, unit?: string): string {
	return unit ? `${text} ${unit}` : text;
}
export function shortHash(h?: string, n = 12): string {
	return h ? `${h.slice(0, n)}…` : '—';
}
