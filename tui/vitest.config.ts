import {defineConfig} from 'vitest/config';

export default defineConfig({
	test: {
		include: ['test/**/*.test.{ts,tsx}'],
		exclude: ['test/e2e/**'], // needs a build and a pty: npm run test:e2e
		environment: 'node',
		testTimeout: 20_000,
	},
});
