import { test as setup, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Real sign-in through the real Clerk widget, using a dedicated test
// account created via Clerk's Backend API (never the developer's own
// account, never a bypass) — see RUNBOOK.md. Saves storage state so the
// rest of the suite doesn't repeat this on every test.
const AUTH_FILE = path.join(__dirname, '.auth/user.json');

setup('reset test-user onboarding state and sign in', async ({ page }) => {
	const email = process.env.PLAYWRIGHT_TEST_EMAIL;
	const password = process.env.PLAYWRIGHT_TEST_PASSWORD;
	if (!email || !password) {
		throw new Error('PLAYWRIGHT_TEST_EMAIL / PLAYWRIGHT_TEST_PASSWORD must be set (see frontend/.env.test.example)');
	}

	const secret = process.env.CLERK_SECRET_KEY;
	if (secret) {
		// Reset onboarding metadata so the onboarding flow is exercised
		// deterministically on every run, not just the first ever run.
		const usersResp = await fetch(`https://api.clerk.com/v1/users?email_address=${encodeURIComponent(email)}`, {
			headers: { Authorization: `Bearer ${secret}` }
		});
		const users = (await usersResp.json()) as Array<{ id: string }>;
		if (users[0]) {
			await fetch(`https://api.clerk.com/v1/users/${users[0].id}/metadata`, {
				method: 'PATCH',
				headers: { Authorization: `Bearer ${secret}`, 'Content-Type': 'application/json' },
				body: JSON.stringify({ unsafe_metadata: {} })
			});
		}
	}

	await page.goto('/sign-in');
	await expect(page.getByText(/synthetic research benchmark/i)).toBeVisible({ timeout: 15_000 });

	await page.getByLabel(/email address/i).fill(email);
	await page.getByRole('button', { name: 'Continue', exact: true }).click();

	await page.getByLabel(/^password$/i).fill(password);
	await page.getByRole('button', { name: 'Continue', exact: true }).click();

	await page.waitForURL(/\/(onboarding|overview)/, { timeout: 20_000 });

	fs.mkdirSync(path.dirname(AUTH_FILE), { recursive: true });
	await page.context().storageState({ path: AUTH_FILE });
});
