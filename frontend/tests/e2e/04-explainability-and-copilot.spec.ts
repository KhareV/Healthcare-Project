import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('AI + SHAP page renders TreeSHAP contributor bars', async ({ page }) => {
	await page.goto('/ai/insights');
	await expect(page.getByText('Make the forecast legible')).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('Increased the output').first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('Decreased the output').first()).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '10_explainability.png'), fullPage: true });
});

test('Trajectory Copilot drawer opens from the dashboard header', async ({ page }) => {
	await page.goto('/patients');
	await expect(page.getByText('Current SOFA')).toBeVisible({ timeout: 15_000 });

	await page.getByRole('button', { name: 'Toggle Trajectory Copilot' }).click();
	await expect(page.getByText('TRAJECTORY COPILOT')).toBeVisible({ timeout: 10_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '11_trajectory_copilot.png'), fullPage: true });
});
