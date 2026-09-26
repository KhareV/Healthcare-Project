import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

// Overrides the project's default (signed-in) storage state with a clean,
// unauthenticated one for this file only.
test.use({ storageState: { cookies: [], origins: [] } });

test('public landing page renders without authentication', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByText('Recovery.')).toBeVisible({ timeout: 15_000 });
	await expect(page.getByText('SYNTHETIC RESEARCH BENCHMARK')).toBeVisible();
	await page.screenshot({ path: path.join(SHOT_DIR, '01_landing.png'), fullPage: true });
});

test('an unauthenticated visitor to a protected route is redirected to sign-in', async ({ page }) => {
	await page.goto('/overview');
	await page.waitForURL(/\/sign-in/, { timeout: 15_000 });
	await expect(page.getByText('SYNTHETIC RESEARCH BENCHMARK · NOT A CLINICAL TOOL')).toBeVisible();
});

test('sign-in page renders the real Clerk widget', async ({ page }) => {
	await page.goto('/sign-in');
	await expect(page.getByLabel(/email address/i)).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '02_sign_in.png'), fullPage: true });
});
