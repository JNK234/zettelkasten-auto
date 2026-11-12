import {Text} from 'ink';
import {PageCheckView} from '../../ui/views/PageView.js';
import {WriteView} from '../../ui/views/WriteView.js';
import {parseLimit, takeFlag, tokenize, type Token} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE =
	'/page new "<title>" @zettel… [from "<query>"]… [limit <n>] [tags "<value>"] [write] · /page check [@page] · ' +
	'/page refresh @page [write] · /page exclude @page @zettel… [undo] [write]';
const WRITE_HINT = 'add write to the command to write it';

function usage(detail?: string): Error {
	return new Error(`Usage: ${USAGE}${detail ? ` (${detail})` : ''}`);
}

function mentions(tokens: Token[]): string[] {
	const words = tokens.filter((token) => token.kind !== 'mention');
	if (words.length > 0) {
		throw usage(`did not understand "${words.map((token) => token.value).join(' ')}"; name notes with @`);
	}

	return tokens.map((token) => token.value);
}

export const page: Command = {
	name: 'page',
	usage: USAGE,
	summary: 'Pages made from zettels: new (a dry run first), check what needs attention, refresh, exclude',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const sub = tokens.shift();
		const write = takeFlag(tokens, 'write', '--write');
		switch (sub?.value) {
			case 'new': {
				const title = tokens.shift();
				if (!title || title.kind !== 'word') {
					throw usage('a title first');
				}

				const zettels: string[] = [];
				const queries: string[] = [];
				let limit: number | undefined;
				let tags: string | undefined;
				for (let index = 0; index < tokens.length; index++) {
					const token = tokens[index]!;
					const value = tokens[index + 1]?.value;
					if (token.kind === 'mention') {
						zettels.push(token.value);
					} else if (['from', 'limit', 'tags'].includes(token.value) && value !== undefined) {
						index++;
						if (token.value === 'from') {
							queries.push(value);
						} else if (token.value === 'limit') {
							limit = parseLimit(value, USAGE);
						} else {
							tags = value;
						}
					} else {
						throw usage(`did not understand "${token.value}"`);
					}
				}

				if (zettels.length === 0 && queries.length === 0) {
					throw usage('name zettels with @, or give from "<query>"');
				}

				const result = await withFormat(async () =>
					api.pageNew({title: title.value, zettels: zettels.length > 0 ? zettels : null, queries: queries.length > 0 ? queries : null, ...(limit ? {limit} : {}), tags: tags ?? null, dry_run: !write}),
				);
				const extra = result.from.map((from) => (
					<Text key={from.query} dimColor>
						from &quot;{from.query}&quot;: {from.taken.length} of {from.zettels} matching zettels ({from.matched} notes matched)
					</Text>
				));
				return <WriteView result={result} writeHint={WRITE_HINT} extra={extra} maxLines={60} />;
			}

			case 'check': {
				const [target, ...rest] = mentions(tokens);
				if (rest.length > 0 || write) {
					throw usage('one page at most');
				}

				const data = await withFormat(async () => api.pageCheck({page: target ?? null}));
				return <PageCheckView data={data} one={target !== undefined} />;
			}

			case 'refresh': {
				const notes = mentions(tokens);
				if (notes.length !== 1) {
					throw usage('one page');
				}

				const result = await withFormat(async () => api.pageRefresh({page: notes[0]!, dry_run: !write}));
				return <WriteView result={result} writeHint={WRITE_HINT} />;
			}

			case 'exclude': {
				const undo = takeFlag(tokens, 'undo', '--undo');
				const [target, ...zettels] = mentions(tokens);
				if (!target || zettels.length === 0) {
					throw usage('a page, then the zettels');
				}

				const result = await withFormat(async () =>
					api.pageExclude({page: target, zettels: zettels as [string, ...string[]], undo, dry_run: !write}),
				);
				return <WriteView result={result} writeHint={WRITE_HINT} />;
			}

			default: {
				throw usage();
			}
		}
	},
};
