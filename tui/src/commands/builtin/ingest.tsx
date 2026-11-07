import {resolve} from 'node:path';
import {homedir} from 'node:os';
import {IngestView} from '../../ui/views/IngestView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/ingest <file.md|.txt|.html> [as <type>] [--dry-run]';

export const ingest: Command = {
	name: 'ingest',
	usage: USAGE,
	summary: 'Bring a file in as a source note; stops and shows the existing note if it is a duplicate',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const dryRun = takeFlag(tokens, '--dry-run', 'dry-run');
		let type = 'source';
		const as = tokens.findIndex((token) => token.kind === 'word' && (token.value === 'as' || token.value === '--as'));
		if (as >= 0) {
			const value = tokens[as + 1]?.value;
			if (!value) {
				throw new Error(`Usage: ${USAGE}`);
			}

			type = value;
			tokens.splice(as, 2);
		}

		if (tokens.length !== 1 || tokens[0]!.kind !== 'word') {
			throw new Error(`Usage: ${USAGE}`);
		}

		const raw = tokens[0]!.value;
		// The server runs elsewhere (repo root): send an absolute path.
		const path = resolve(raw.startsWith('~/') ? `${homedir()}${raw.slice(1)}` : raw);
		const result = await withFormat(async () => api.ingest({path, type, dry_run: dryRun}));
		return <IngestView result={result} />;
	},
};
