import {CoverageView} from '../../ui/views/CoverageView.js';
import {parseLimit, takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/coverage [all | <per list>]';

export const coverage: Command = {
	name: 'coverage',
	usage: USAGE,
	summary: 'Sources nothing links to, and notes that cite no source (and why)',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const all = takeFlag(tokens, 'all');
		if (tokens.length > 1) {
			throw new Error(`Usage: ${USAGE}`);
		}

		const perGroup = all ? undefined : (parseLimit(tokens[0]?.value, USAGE) ?? (tokens[0] ? undefined : 5));
		const data = await withFormat(async () => api.coverage({limit: null}));
		return <CoverageView data={data} perGroup={perGroup} />;
	},
};
