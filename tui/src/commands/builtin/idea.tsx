import {WriteView} from '../../ui/views/WriteView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/idea new <finding-id> ["<title>"] [tags "<value>"] [write] · /idea set @idea <open | dropped | done> [write]';
const WRITE_HINT = 'add write to the command to write it';

export const idea: Command = {
	name: 'idea',
	usage: USAGE,
	summary: 'An idea note from a finding (/findings gives the ids), or set an idea\'s status',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const sub = tokens.shift();
		const write = takeFlag(tokens, 'write', '--write');
		if (sub?.value === 'new') {
			const finding = tokens.shift();
			if (!finding || finding.kind !== 'word') {
				throw new Error(`Usage: ${USAGE}`);
			}

			let title: string | undefined;
			let tags: string | undefined;
			for (let index = 0; index < tokens.length; index++) {
				const token = tokens[index]!;
				if (token.value === 'tags' && tokens[index + 1]) {
					tags = tokens[++index]!.value;
				} else if (title === undefined && token.kind === 'word') {
					title = token.value;
				} else {
					throw new Error(`Usage: ${USAGE} (did not understand "${token.value}")`);
				}
			}

			const result = await withFormat(async () => api.ideaNew({finding: finding.value, title: title ?? null, tags: tags ?? null, dry_run: !write}));
			return <WriteView result={result} writeHint={WRITE_HINT} maxLines={60} />;
		}

		if (sub?.value === 'set') {
			const [note, status, ...rest] = tokens;
			if (!note || note.kind !== 'mention' || !status || status.kind !== 'word' || rest.length > 0) {
				throw new Error(`Usage: ${USAGE}`);
			}

			const result = await withFormat(async () => api.ideaSet({idea: note.value, status: status.value, dry_run: !write}));
			return <WriteView result={result} writeHint={WRITE_HINT} />;
		}

		throw new Error(`Usage: ${USAGE}`);
	},
};
