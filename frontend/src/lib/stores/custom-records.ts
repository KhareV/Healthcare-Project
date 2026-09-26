// Session-local memory of custom records this browser has created, so
// Patient Replay can list them without a backend list endpoint (the backend
// intentionally keeps no per-user index — see serving/v2/custom_record.py).
// Purely a convenience: the source of truth is always the server's in-memory
// ephemeral registry, which this only ever mirrors, never replaces.
const STORAGE_KEY = 'prt-custom-records-v1';

export type RememberedCustomRecord = { stay_id: string; patient_alias: string };

function readAll(): RememberedCustomRecord[] {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		return raw ? (JSON.parse(raw) as RememberedCustomRecord[]) : [];
	} catch {
		return [];
	}
}

export function listRememberedCustomRecords(): RememberedCustomRecord[] {
	return readAll();
}

export function rememberCustomRecord(record: RememberedCustomRecord): void {
	try {
		const all = readAll().filter((r) => r.stay_id !== record.stay_id);
		all.unshift(record);
		localStorage.setItem(STORAGE_KEY, JSON.stringify(all.slice(0, 20)));
	} catch {
		/* localStorage unavailable (private mode, etc.) — non-fatal */
	}
}

export function forgetCustomRecord(stayId: string): void {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(readAll().filter((r) => r.stay_id !== stayId)));
	} catch {
		/* ignore */
	}
}
