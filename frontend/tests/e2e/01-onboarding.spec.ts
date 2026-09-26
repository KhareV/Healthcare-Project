import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

// Runs first (file name ordering, workers=1): global.setup.ts's real Clerk
// sign-in redirected here because onboarding metadata was reset. Completes
// the real onboarding flow so every later spec file can assume it's done.
test('completes the real onboarding flow (role -> disclaimer -> data source -> done)', async ({ page }) => {
	await page.goto('/onboarding');
	await expect(page.getByText('Personalized Patient Recovery Trajectory')).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '03_onboarding_welcome.png'), fullPage: true });

	await page.getByRole('button', { name: 'Continue' }).click();

	await page.getByText('Clinical Researcher').click();
	await page.getByRole('button', { name: 'Continue' }).click();

	await page.getByRole('checkbox').check();
	await page.getByRole('button', { name: 'Continue' }).click();

	await page.getByText('Explore Demo Patients').click();
	await page.getByRole('button', { name: 'Continue' }).click();

	await expect(page.getByText("You're set up.")).toBeVisible();
	await page.getByRole('button', { name: /enter workspace/i }).click();

	await page.waitForURL(/\/overview/, { timeout: 15_000 });
	await expect(page.getByText('Welcome,')).toBeVisible();
});
