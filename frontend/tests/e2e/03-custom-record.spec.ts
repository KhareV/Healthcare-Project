import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('enter my own record: fill the form, submit, and open a real prediction for it', async ({ page }) => {
	await page.goto('/patients/custom');
	await expect(page.getByText('Build a synthetic ICU episode and forecast it.')).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '07_custom_record_entry.png'), fullPage: true });

	await page.getByLabel('Patient alias').fill('Playwright E2E Patient');
	await page.getByRole('button', { name: 'Stable recovery' }).click();

	await page.getByRole('button', { name: 'Build my record' }).click();

	await expect(page.getByText('Playwright E2E Patient')).toBeVisible({ timeout: 20_000 });
	await expect(page.getByText('DATA READINESS')).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '08_custom_record_readiness.png'), fullPage: true });

	await page.getByRole('button', { name: /open in patient replay/i }).click();
	await page.waitForURL(/\/patients\?stay_id=CUSTOM-/, { timeout: 15_000 });

	// .first(): with persistence enabled, repeated runs accumulate more than
	// one same-named custom record (and "YOUR RECORD" badge) over time --
	// this assertion only cares that the one just created renders correctly.
	await expect(page.getByText('Playwright E2E Patient').first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('Current SOFA')).toBeVisible();
	await expect(page.getByText('YOUR RECORD').first()).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '09_custom_record_prediction.png'), fullPage: true });
});
