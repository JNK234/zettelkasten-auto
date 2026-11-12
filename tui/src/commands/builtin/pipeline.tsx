import {PipelineView} from '../../ui/views/PipelineView.js';
import {parseLimit, takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/pipeline [all | <rows>]';

export const pipeline: Command = {
	name: 'pipeline',
	usage: USAGE,
	summary: 'Each source: ingested, extracted (a zettel links it) or compiled (a page uses it); and the gaps',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const all = takeFlag(tokens, 'all');
		if (tokens.length > 1) {
			throw new Error(`Usage: ${USAGE}`);
		}

		const limit = all ? undefined : (parseLimit(tokens[0]?.value, USAGE) ?? (tokens[0] ? undefined : 8));
		const data = await withFormat(async () => api.pipeline({limit: limit ?? null}));
		return <PipelineView data={data} />;
	},
};
