import { redirect } from '@sveltejs/kit';

// Retired inherited template routes (ECG / federated learning / device monitoring)
// are not part of this healthcare product. Old URLs land on the closest active page.
const REDIRECTS: [string, string][] = [
	['/fl', '/research/analytics'],
	['/ai/confidence', '/research/results'],
	['/ai/anomaly', '/ai/insights'],
	['/ai/baseline', '/ai/insights'],
	['/ai/explainability', '/ai/insights'],
	['/ai/model', '/ai/insights'],
	['/signals', '/overview'],
	['/monitoring', '/overview'],
	['/monitor', '/overview'],
	['/alerts', '/overview'],
	['/system/devices', '/system/data'],
	['/system/notifications', '/system/data'],
	['/system/settings', '/system/data'],
	['/system/health', '/system/data']
];

export function load({ url }) {
	const path = url.pathname.replace(/\/+$/, '') || '/';
	for (const [prefix, target] of REDIRECTS) {
		if (path === prefix || path.startsWith(prefix + '/')) throw redirect(307, target);
	}
}
