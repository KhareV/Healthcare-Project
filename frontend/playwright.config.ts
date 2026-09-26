import { defineConfig, devices } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Minimal .env.test loader (no extra dependency): PLAYWRIGHT_TEST_EMAIL,
// PLAYWRIGHT_TEST_PASSWORD, CLERK_SECRET_KEY. See .env.test.example.
const envTestPath = path.resolve(__dirname, '.env.test');
if (fs.existsSync(envTestPath)) {
	for (const line of fs.readFileSync(envTestPath, 'utf-8').split('\n')) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith('#')) continue;
		const eq = trimmed.indexOf('=');
		if (eq === -1) continue;
		const key = trimmed.slice(0, eq).trim();
		const value = trimmed.slice(eq + 1).trim();
		if (!(key in process.env)) process.env[key] = value;
	}
}

// Browser verification for the current SvelteKit UI. Runs against the real
// V2 API (port 8010) and the real Vite dev server (port 5173) — no mocked
// backend. Authentication uses a dedicated Clerk test account (created via
// Clerk's Backend API for this purpose, never the developer's own account)
// signing in through the real Clerk widget — not a bypass — so this proves
// the actual production auth path works, per the project's rule that a test
// mode must never weaken production Clerk behavior.
export default defineConfig({
	testDir: './tests/e2e',
	timeout: 45_000,
	expect: { timeout: 10_000 },
	fullyParallel: false, // shared signed-in storage state; keep workers sequential per project
	workers: 1,
	retries: 0,
	reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
	use: {
		baseURL: 'http://localhost:5173',
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	projects: [
		{ name: 'setup', testMatch: /global\.setup\.ts/ },
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'], storageState: 'tests/e2e/.auth/user.json' },
			dependencies: ['setup']
		}
	],
	webServer: [
		{
			command:
				'cd .. && PYTHONPATH=src:. python3 -m uvicorn api.v2_app:app --factory --host 127.0.0.1 --port 8010',
			url: 'http://127.0.0.1:8010/health',
			reuseExistingServer: true,
			timeout: 60_000
		},
		{
			command: 'npm run dev',
			url: 'http://localhost:5173',
			reuseExistingServer: true,
			timeout: 60_000
		}
	]
});
