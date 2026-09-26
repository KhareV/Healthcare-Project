import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('Model Performance page renders frozen final-evaluation metrics', async ({ page }) => {
	await page.goto('/research/analytics');
	await expect(page.getByText(/recovery.*24/i).first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('1.074').first()).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '12_model_performance.png'), fullPage: true });
});

test('Data Quality & Provenance page renders', async ({ page }) => {
	await page.goto('/system/data');
	await expect(page.getByText(/data quality/i).first()).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '13_data_provenance.png'), fullPage: true });
});
