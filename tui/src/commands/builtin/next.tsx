import {NextView} from '../../ui/views/FindingsView.js';
import {parseLimit, takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/next [all | <items>]';

export const next: Command = {
	name: 'next',
	usage: USAGE,
	summary: 'What needs doing, most trusted first: each item with why and the command that resolves it',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const all = takeFlag(tokens, 'all');
		if (tokens.length > 1) {
			throw new Error(`Usage: ${USAGE}`);
		}

		const limit = all ? undefined : (parseLimit(tokens[0]?.value, USAGE) ?? (tokens[0] ? undefined : 10));
		const data = await withFormat(async () => api.next({limit: limit ?? null}));
		return <NextView data={data} />;
	},
};
