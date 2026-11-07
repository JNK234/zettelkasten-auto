import {CheckView} from '../../ui/views/CheckView.js';
import {parseLimit, takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/check [changed] [all | <notes>]';

export const check: Command = {
	name: 'check',
	usage: USAGE,
	summary: 'Violations of the format, grouped by note, with line numbers',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const changed = takeFlag(tokens, 'changed', '--changed');
		const all = takeFlag(tokens, 'all');
		if (tokens.length > 1) {
			throw new Error(`Usage: ${USAGE}`);
		}

		const limit = all ? undefined : (parseLimit(tokens[0]?.value, USAGE) ?? (tokens[0] ? undefined : 8));
		const data = await withFormat(async () => api.check({changed, limit: limit ?? null}));
		return <CheckView data={data} />;
	},
};
