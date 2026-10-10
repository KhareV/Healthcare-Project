import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const API = 'http://127.0.0.1:8010';

test('all eight authorized scenarios run the frozen V3 model at a legal cutoff', async ({ request }) => {
	const response = await request.get(`${API}/showcase/scenarios`);
	expect(response.ok()).toBeTruthy();
	const catalog = await response.json();
	expect(catalog.scenarios).toHaveLength(8);
	for (const scenario of catalog.scenarios) {
		expect(scenario.registered_for_inference).toBe(true);
		expect(scenario.source_cohort).not.toMatch(/(?:final|test|SYN-V3-T)/i);
		expect(scenario.legal_cutoffs.length).toBeGreaterThan(1);
		const prediction = await request.post(`${API}/predict`, {
			data: { stay_id: scenario.stay_id, prediction_time: scenario.legal_cutoffs[0] }
		});
		expect(prediction.ok(), scenario.scenario_id).toBeTruthy();
		const body = await prediction.json();
		expect(body.model_bundle.release).toBe('V3');
		expect(Number.isFinite(body.current_sofa)).toBeTruthy();
	}
});

test('scenario replay, story and two-patient comparison use the real product API', async ({ page }) => {
	await page.goto('/demo');
	await expect(page.getByTestId('scenario-card')).toHaveCount(11);
	await page.getByTestId('scenario-card').first().getByRole('button', { name: /open \/ replay/i }).click();
	await expect(page.getByTestId('metrics')).toBeVisible({ timeout: 20_000 });
	await expect(page.getByTestId('cutoff-label')).toContainText('t0');
	await page.getByRole('button', { name: 'Next cutoff' }).click();
	await expect(page.getByTestId('cutoff-label')).toContainText('t1');
	await page.getByRole('button', { name: 'STORY' }).click();
	await page.getByRole('button', { name: 'COMPARE' }).click();
	await expect(page.getByRole('region', { name: 'Comparison controls' })).toBeVisible();
});

test('cohort and model research pages load artifact-backed views at mobile width', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	for (const route of ['/cohort', '/research/analytics', '/research/comparison', '/research/robustness', '/research/diagnostic']) {
		await page.goto(route);
		await expect(page.locator('main')).toBeVisible({ timeout: 20_000 });
		await expect(page.locator('body')).not.toContainText('Could not load');
		const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
		expect(overflow, route).toBeLessThanOrEqual(1);
	}
	await expect(page.getByText(/privileged/i).first()).toBeVisible();
});

test('research report and chart source exports contain the saved V3 evidence identity', async ({ page }) => {
	await page.goto('/reports');
	const reportPromise = page.waitForEvent('download');
	await page.getByRole('button', { name: 'MARKDOWN (.md)' }).click();
	const report = await reportPromise;
	expect(report.suggestedFilename()).toMatch(/prt-research-report-.*\.md/);
	const reportText = await fs.readFile(await report.path(), 'utf8');
	expect(reportText).toContain('V3');
	expect(reportText).toMatch(/source|sha256/i);

	await page.goto('/cohort');
	const sourcePromise = page.waitForEvent('download');
	await page.getByRole('button', { name: /Download Cardiac subtype.*data as CSV/i }).first().click();
	const source = await sourcePromise;
	expect(source.suggestedFilename()).toMatch(/\.csv$/);
	const csv = await fs.readFile(await source.path(), 'utf8');
	expect(csv.split('\n').length).toBeGreaterThan(2);
});

test('research charts support keyboard expansion and focus restoration', async ({ page }) => {
	await page.goto('/cohort');
	const expand = page.getByRole('button', { name: /Expand Cardiac subtype/i }).first();
	await expand.focus();
	await page.keyboard.press('Enter');
	const dialog = page.getByRole('dialog', { name: /Cardiac subtype.*expanded/i });
	await expect(dialog).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(dialog).toBeHidden();
	await expect(expand).toBeFocused();
});
