import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

// A tiny, synthetic in-memory PDF fixture -- never a real/sensitive file.
const FAKE_PDF = Buffer.from('%PDF-1.4\n%playwright-e2e-fixture\n');

test('My Health Record: overview, profile edit, conditions, encounters, vitals, reports, and prediction history', async ({ page }) => {
	// Create a custom record and open it in Patient Replay so at least one
	// real prediction gets persisted for its history, and it appears under
	// Encounters/Vitals & Labs.
	await page.goto('/patients/custom');
	await page.getByLabel('Patient alias').fill('Health Record E2E Patient');
	await page.getByRole('button', { name: 'Stable recovery' }).click();
	await page.getByRole('button', { name: 'Build my record' }).click();
	await expect(page.getByText('DATA READINESS')).toBeVisible({ timeout: 20_000 });
	await page.getByRole('button', { name: /open in patient replay/i }).click();
	await page.waitForURL(/\/patients\?stay_id=CUSTOM-/, { timeout: 15_000 });
	await expect(page.getByText('Current SOFA')).toBeVisible({ timeout: 15_000 });

	await page.goto('/health-record');
	await expect(page.getByText('One consolidated, longitudinal research record.')).toBeVisible({ timeout: 15_000 });

	// -- Overview: cards + profile edit --------------------------------
	await expect(page.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
	await page.screenshot({ path: path.join(SHOT_DIR, '14_health_record_overview.png'), fullPage: true });

	const displayNameInput = page.getByLabel('Display name');
	await displayNameInput.fill('Playwright Profile Name');
	await page.getByRole('button', { name: 'Save profile' }).click();
	await expect(displayNameInput).toHaveValue('Playwright Profile Name', { timeout: 10_000 });
	// Persisted, not just local state: a reload still shows it.
	await page.reload();
	await expect(page.getByLabel('Display name')).toHaveValue('Playwright Profile Name', { timeout: 15_000 });

	// -- Conditions: add, edit, list ------------------------------------
	await page.getByRole('tab', { name: 'Conditions' }).click();
	await page.getByPlaceholder('e.g. Ischemic heart disease').fill('Playwright E2E Condition');
	await page.getByRole('button', { name: /add condition/i }).click();
	await expect(page.getByText('Playwright E2E Condition')).toBeVisible({ timeout: 15_000 });

	await page.locator('.condition-list li', { hasText: 'Playwright E2E Condition' }).getByRole('button', { name: 'Edit' }).click();
	await page.locator('.condition-edit-form select').selectOption('resolved');
	await page.locator('.condition-edit-form').getByRole('button', { name: 'Save' }).click();
	await expect(page.locator('.condition-list li', { hasText: 'Playwright E2E Condition' })).toContainText('resolved', { timeout: 10_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '15_health_record_conditions.png'), fullPage: true });

	// -- Encounters ------------------------------------------------------
	await page.getByRole('tab', { name: 'Encounters' }).click();
	await expect(page.locator('.row--encounters', { hasText: 'Health Record E2E Patient' }).last()).toBeVisible({ timeout: 15_000 });

	// -- Vitals & Labs + Support history ---------------------------------
	await page.getByRole('tab', { name: 'Vitals & Labs' }).click();
	await expect(page.locator('.row--vitals').first()).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '16_health_record_vitals.png'), fullPage: true });

	// -- Reports: upload, visible, download, delete ----------------------
	await page.getByRole('tab', { name: 'Reports' }).click();
	await page.getByPlaceholder('Title').fill('Playwright E2E Report');
	await page.setInputFiles('input[type="file"]', { name: 'e2e-fixture.pdf', mimeType: 'application/pdf', buffer: FAKE_PDF });
	await page.getByRole('button', { name: /^upload$/i }).click();
	const reportCard = page.locator('.report-card', { hasText: 'Playwright E2E Report' });
	await expect(reportCard).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '17_health_record_reports.png'), fullPage: true });

	const downloadPromise = page.waitForEvent('download');
	await reportCard.getByRole('button', { name: 'Download report' }).click();
	const download = await downloadPromise;
	expect(download.suggestedFilename()).toBe('e2e-fixture.pdf');

	await reportCard.getByRole('button', { name: 'Delete report' }).click();
	await expect(reportCard).not.toBeVisible({ timeout: 15_000 });

	// -- Prediction history ------------------------------------------------
	await page.getByRole('tab', { name: 'Prediction History' }).click();
	await expect(page.locator('.row--predictions').first()).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '18_health_record_predictions.png'), fullPage: true });

	// -- Cleanup: remove the condition created by this test ----------------
	await page.getByRole('tab', { name: 'Conditions' }).click();
	await page.locator('.condition-list li', { hasText: 'Playwright E2E Condition' }).getByRole('button', { name: 'Remove condition' }).click();
	await expect(page.getByText('Playwright E2E Condition')).not.toBeVisible({ timeout: 15_000 });
});
