import {FixView} from '../../ui/views/FixView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/fix [apply] [mirrors | headings] [all]';

export const fix: Command = {
	name: 'fix',
	usage: USAGE,
	summary: 'Preview the safe repairs (missing mirrors, missing headings) as diffs; /fix apply writes them',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const apply = takeFlag(tokens, 'apply', '--apply');
		const all = takeFlag(tokens, 'all');
		type Kind = 'missing-mirror' | 'missing-heading';
		const only: Kind[] = [];
		if (takeFlag(tokens, 'mirrors', 'missing-mirror')) {
			only.push('missing-mirror');
		}

		if (takeFlag(tokens, 'headings', 'missing-heading')) {
			only.push('missing-heading');
		}

		if (tokens.length > 0) {
			throw new Error(`Usage: ${USAGE} (unknown "${tokens.map((token) => token.value).join(' ')}")`);
		}

		const kinds = only.length > 0 ? (only as [Kind, ...Kind[]]) : null;
		const result = await withFormat(async () => api.fix({dry_run: !apply, only: kinds}));
		return <FixView result={result} maxDiffs={all ? Number.POSITIVE_INFINITY : 3} />;
	},
};
