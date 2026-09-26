import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('workspace home shows demo-patient and custom-record entry points', async ({ page }) => {
	await page.goto('/overview');
	await expect(page.getByText('Explore Demo Patients')).toBeVisible();
	await expect(page.getByText('Enter My Own Record')).toBeVisible();
	await expect(page.getByText('Connect EHR Sandbox')).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '04_workspace.png'), fullPage: true });
});

test('demo-patient list renders three aliased patients', async ({ page }) => {
	await page.goto('/patients');
	await expect(page.getByText('DEMO-CARDIAC-001').first()).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('DEMO-CARDIAC-002').first()).toBeVisible();
	await expect(page.getByText('DEMO-CARDIAC-003').first()).toBeVisible();
});

test('patient replay: prediction cards render and cutoff navigation advances the forecast', async ({ page }) => {
	await page.goto('/patients');
	await expect(page.getByText('Current SOFA')).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('Predicted SOFA +24h')).toBeVisible();
	await expect(page.getByText('Remaining ICU stay time')).toBeVisible();
	await expect(page.getByText('New organ-support risk (24h)')).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '05_demo_patient_replay.png'), fullPage: true });

	const cutoffCountBefore = await page.locator('.cutoff-count').textContent();
	await page.getByRole('button', { name: /^Next/ }).click();
	await expect(page.locator('.cutoff-count')).not.toHaveText(cutoffCountBefore ?? '', { timeout: 10_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '06_recovery_trajectory.png'), fullPage: true });
});
