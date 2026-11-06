import {HealthView} from '../../ui/views/HealthView.js';
import type {Command} from '../registry.js';

export const health: Command = {
	name: 'health',
	usage: '/health [limit]',
	summary: 'Broken links, orphans, dead ends, duplicates, empty notes',
	async run(input, {api}) {
		const limit = input.text ? Number.parseInt(input.text, 10) : 5;
		if (!Number.isInteger(limit) || limit < 1) {
			throw new Error(`Usage: /health [limit], where limit is a positive number (got "${input.text}")`);
		}

		return <HealthView data={await api.health({limit})} />;
	},
};
