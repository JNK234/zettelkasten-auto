import {WriteView} from '../../ui/views/WriteView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/link @note <relation> @note [--dry-run]';

export const link: Command = {
	name: 'link',
	usage: USAGE,
	summary: 'Relate two notes (e.g. supports) and write the mirror (supported-by) in the other',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const dryRun = takeFlag(tokens, '--dry-run', 'dry-run');
		takeFlag(tokens, 'as', '--as');
		const notes = tokens.filter((token) => token.kind === 'mention');
		const words = tokens.filter((token) => token.kind === 'word');
		if (notes.length !== 2 || words.length !== 1) {
			throw new Error(`Usage: ${USAGE}, e.g. /link @EVD… supports @CLM… (type @ to pick notes)`);
		}

		const result = await withFormat(async () =>
			api.link({source: notes[0]!.value, target: notes[1]!.value, relation: words[0]!.value, dry_run: dryRun}),
		);
		return <WriteView result={result} />;
	},
};
