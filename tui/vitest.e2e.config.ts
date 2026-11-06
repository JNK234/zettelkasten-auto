import {defineConfig} from 'vitest/config';

// End-to-end tests: the built CLI (dist/cli.js) in a real pseudo-terminal.
// Run with `npm run test:e2e`, which builds first.
export default defineConfig({
	test: {
		include: ['test/e2e/**/*.test.ts'],
		environment: 'node',
		testTimeout: 90_000,
		hookTimeout: 30_000,
	},
});
