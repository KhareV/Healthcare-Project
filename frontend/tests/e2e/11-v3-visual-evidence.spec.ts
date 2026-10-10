import { test, expect } from '@playwright/test';
import path from 'node:path';
import { SHOT_DIR } from './shot-dir';

test('capture the actual V3 research platform at desktop, tablet and mobile sizes', async ({ page }) => {
	const views = [
		{ route: '/cohort', name: 'cohort' },
		{ route: '/research/analytics', name: 'model_research' },
		{ route: '/research/diagnostic', name: 'latent_diagnostic' },
		{ route: '/demo', name: 'scenario_studio' }
	];
	const sizes = [
		{ width: 1440, height: 900, name: 'desktop' },
		{ width: 768, height: 1024, name: 'tablet' },
		{ width: 390, height: 844, name: 'mobile' }
	];
	for (const size of sizes) {
		await page.setViewportSize({ width: size.width, height: size.height });
		for (const view of views) {
			await page.goto(view.route);
			if (view.name === 'scenario_studio') {
				await expect(page.getByRole('heading', { name: 'Scenario Studio' })).toBeVisible({ timeout: 20_000 });
				await expect(page.getByTestId('scenario-card')).toHaveCount(11);
			} else {
				await expect(page.locator('main')).toBeVisible({ timeout: 20_000 });
				await expect(page.locator('main svg').first()).toBeVisible({ timeout: 20_000 });
			}
			await page.screenshot({ path: path.join(SHOT_DIR, `v3_${view.name}_${size.name}.png`), fullPage: true });
			const layout = await page.evaluate(() => ({
				overflow: document.documentElement.scrollWidth - innerWidth,
				offenders: [...document.querySelectorAll<HTMLElement>('*')]
					.filter((element) => element.getBoundingClientRect().right > innerWidth + 1)
					.slice(0, 5).map((element) => `${element.tagName}.${element.className} right=${Math.round(element.getBoundingClientRect().right)} text=${element.textContent?.slice(0, 100)}`)
			}));
			expect(layout.overflow, `${view.name} at ${size.name}: ${layout.offenders.join(', ')}`).toBeLessThanOrEqual(1);
		}
	}
});
