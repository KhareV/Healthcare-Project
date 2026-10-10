import { test, expect } from '@playwright/test';

const API = 'http://127.0.0.1:8010';

test('API serves the V3 release and reports its bundle', async ({ request }) => {
	const health = await (await request.get(`${API}/health`)).json();
	expect(health.release).toBe('V3');
	expect(health.model_families.recovery48).toBe('catboost');
	expect(health.calibrator).toBe('platt');
});

test('a demo prediction carries the V3 bundle identity and no V2 metadata', async ({ request }) => {
	const demo = await (await request.get(`${API}/demo-subjects`)).json();
	const s = demo.demo_subjects[0];
	const res = await request.post(`${API}/predict`, { data: { stay_id: s.stay_id, prediction_time: s.legal_cutoffs[s.legal_cutoffs.length - 1] } });
	expect(res.ok()).toBeTruthy();
	const p = await res.json();
	expect(p.model_bundle.release).toBe('V3');
	expect(JSON.stringify(p.versions)).not.toMatch(/v2/i);
	expect(Number.isFinite(p.current_sofa)).toBeTruthy();
});

test('data provenance page shows the V3 bundle badge', async ({ page }) => {
	await page.goto('/system/data');
	await expect(page.getByTestId('bundle-badge')).toContainText('V3', { timeout: 20_000 });
});

for (const route of ['/overview', '/patients']) {
	test(`primary page ${route} makes no V2 claim`, async ({ page }) => {
		await page.goto(route);
		await expect(page.getByTestId('shell-bundle')).toContainText('V3', { timeout: 20_000 });
		await expect(page.locator('body')).not.toContainText(/\bV2\b/);
	});
}

for (const [from, to] of [['/fl/overview', /\/research\/analytics/], ['/signals/ecg', /\/overview/], ['/monitor', /\/overview/], ['/system/devices', /\/system\/data/]] as const) {
	test(`retired route ${from} redirects`, async ({ page }) => {
		await page.goto(from);
		await page.waitForURL(to, { timeout: 15_000 });
	});
}
