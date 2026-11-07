import {WriteView} from '../../ui/views/WriteView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/new <type> "<title>" [Key=Value]… [<relation> @note]… [--dry-run]';

export const newNote: Command = {
	name: 'new',
	usage: USAGE,
	summary: 'Create a note of a type from the format (refused if it would fail the check)',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const dryRun = takeFlag(tokens, '--dry-run', 'dry-run');
		const [type, title, ...rest] = tokens;
		if (!type || type.kind !== 'word' || !title || title.kind !== 'word') {
			throw new Error(`Usage: ${USAGE}`);
		}

		const values: Record<string, string> = {};
		const relations: Array<{relation: string; note: string}> = [];
		for (let index = 0; index < rest.length; index++) {
			const token = rest[index]!;
			const next = rest[index + 1];
			if (token.kind === 'word' && next?.kind === 'mention') {
				relations.push({relation: token.value, note: next.value});
				index++;
			} else if (token.kind === 'word' && token.value.includes('=')) {
				const at = token.value.indexOf('=');
				values[token.value.slice(0, at)] = token.value.slice(at + 1);
			} else {
				throw new Error(`Usage: ${USAGE} (did not understand "${token.kind === 'mention' ? '@' : ''}${token.value}")`);
			}
		}

		const result = await withFormat(async () => api.newNote({type: type.value, title: title.value, values, relations, dry_run: dryRun}));
		return <WriteView result={result} />;
	},
};
