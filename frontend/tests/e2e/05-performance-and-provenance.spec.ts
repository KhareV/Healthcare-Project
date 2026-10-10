import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('Model Research page renders the frozen V3 final-evaluation metrics', async ({ page }) => {
	await page.goto('/research/analytics');
	await expect(page.getByText(/recovery.*24/i).first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText(/1\.026/).first()).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '12_model_performance.png'), fullPage: true });
});

test('Data Quality & Provenance page renders', async ({ page }) => {
	await page.goto('/system/data');
	await expect(page.getByText(/data quality/i).first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByTestId('bundle-badge')).toContainText('V3', { timeout: 20_000 });
	await expect(page.getByText(/HISTORICAL V2 \/ NOT SERVED/)).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '13_data_provenance.png'), fullPage: true });
});
