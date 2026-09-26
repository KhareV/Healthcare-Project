import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('My Health Record: a created custom record and its prediction history are visible, and conditions can be added and removed', async ({ page }) => {
	// Create a custom record and open it in Patient Replay so at least one
	// real prediction gets persisted for its history.
	await page.goto('/patients/custom');
	await page.getByLabel('Patient alias').fill('Health Record E2E Patient');
	await page.getByRole('button', { name: 'Stable recovery' }).click();
	await page.getByRole('button', { name: 'Build my record' }).click();
	await expect(page.getByText('DATA READINESS')).toBeVisible({ timeout: 20_000 });
	await page.getByRole('button', { name: /open in patient replay/i }).click();
	await page.waitForURL(/\/patients\?stay_id=CUSTOM-/, { timeout: 15_000 });
	await expect(page.getByText('Current SOFA')).toBeVisible({ timeout: 15_000 });

	await page.goto('/health-record');
	await expect(page.getByText('Your conditions and custom records, in one place.')).toBeVisible({ timeout: 15_000 });

	// The record just created is durable (or at least visible for this
	// process's lifetime) via the real backend list -- not client-side
	// bookkeeping. .last(): the list is sorted oldest-first, and repeated
	// runs against a persisted backend accumulate more than one same-named
	// record over time (this alias is reused every run) -- .last() is the
	// one THIS run just created and just replayed a cutoff for.
	const recordCard = page.locator('.record-card', { hasText: 'Health Record E2E Patient' }).last();
	await expect(recordCard).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '14_health_record.png'), fullPage: true });

	await recordCard.getByRole('button', { name: /prediction history/i }).click();
	await expect(recordCard.locator('.history-row').first()).toBeVisible({ timeout: 15_000 });
	await page.screenshot({ path: path.join(SHOT_DIR, '15_health_record_prediction_history.png'), fullPage: true });

	// Conditions: pure display context, added/removed via the real backend.
	await page.getByPlaceholder('e.g. Ischemic heart disease').fill('Playwright E2E Condition');
	await page.getByRole('button', { name: /add condition/i }).click();
	await expect(page.getByText('Playwright E2E Condition')).toBeVisible({ timeout: 15_000 });

	await page.locator('.condition-list li', { hasText: 'Playwright E2E Condition' }).getByRole('button', { name: 'Remove condition' }).click();
	await expect(page.getByText('Playwright E2E Condition')).not.toBeVisible({ timeout: 15_000 });
});
