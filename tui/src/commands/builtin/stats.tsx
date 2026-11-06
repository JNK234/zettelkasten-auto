import {StatsView} from '../../ui/views/StatsView.js';
import type {Command} from '../registry.js';

export const stats: Command = {
	name: 'stats',
	summary: 'Notes, links, tags and the most linked / most wanted notes',
	async run(_input, {api}) {
		return <StatsView data={await api.stats({top: 5})} />;
	},
};
