import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SHOT_DIR } from './shot-dir';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_PDF = fs.readFileSync(path.join(__dirname, 'fixtures', 'synthetic-lab-report.pdf'));

test('Report Intelligence: analyze a report, review candidates, and confirm selected measurements into an encounter', async ({ page }) => {
	// A fresh encounter to confirm measurements into.
	await page.goto('/patients/custom');
	await page.getByLabel('Patient alias').fill('Report Intelligence E2E Patient');
	await page.getByRole('button', { name: 'Stable recovery' }).click();
	await page.getByRole('button', { name: 'Build my record' }).click();
	await expect(page.getByText('DATA READINESS')).toBeVisible({ timeout: 20_000 });

	await page.goto('/health-record');
	await page.getByRole('tab', { name: 'Reports' }).click();

	await page.getByPlaceholder('Title').fill('Report Intelligence E2E Report');
	await page.setInputFiles('input[type="file"]', { name: 'synthetic-lab-report.pdf', mimeType: 'application/pdf', buffer: FIXTURE_PDF });
	await page.getByRole('button', { name: /^upload$/i }).click();
	const reportCard = page.locator('.report-card', { hasText: 'Report Intelligence E2E Report' });
	await expect(reportCard).toBeVisible({ timeout: 15_000 });

	// A real Groq call -- this project's testing philosophy insists on
	// proving the actual extraction, not a mocked stand-in.
	await reportCard.getByRole('button', { name: /^analyze$/i }).click();
	await expect(page.getByText('REPORT ANALYSIS')).toBeVisible({ timeout: 30_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '19_report_analysis_candidates.png'), fullPage: true });

	const candidateRows = page.locator('.candidate-row');
	await expect(candidateRows.first()).toBeVisible({ timeout: 10_000 });
	// The fixture PDF names three measurements this product's canonical
	// vocabulary supports -- expect at least one to be selectable.
	await expect(page.locator('.candidate-row:not(.candidate-row--disabled)').first()).toBeVisible({ timeout: 10_000 });

	await page.locator('.candidate-confirm-row select').selectOption({ label: 'Report Intelligence E2E Patient' });
	await page.getByLabel('Hours since admission').fill('6');
	await page.getByRole('button', { name: /add \d+ selected measurement/i }).click();

	// A successful confirm re-loads the page's data; the report card and its
	// analysis panel should still be present afterward.
	await expect(page.locator('.report-card', { hasText: 'Report Intelligence E2E Report' })).toBeVisible({ timeout: 20_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '20_report_measurements_confirmed.png'), fullPage: true });

	// The confirmed measurement(s) now show up in the Vitals & Labs history.
	await page.getByRole('tab', { name: 'Vitals & Labs' }).click();
	await expect(page.locator('.row--vitals', { hasText: 'Report Intelligence E2E Patient' }).first()).toBeVisible({ timeout: 15_000 });

	// Clean up: remove the report this test created.
	await page.getByRole('tab', { name: 'Reports' }).click();
	await page.locator('.report-card', { hasText: 'Report Intelligence E2E Report' }).getByRole('button', { name: 'Delete report' }).click();
	await expect(page.locator('.report-card', { hasText: 'Report Intelligence E2E Report' })).not.toBeVisible({ timeout: 15_000 });
});
