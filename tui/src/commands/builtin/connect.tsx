import {RpcError} from '../../rpc/client.js';
import type {VaultApi} from '../../rpc/api.js';
import {LinksView, NotesToWriteView, RelatedView, UnlinkedView} from '../../ui/views/ConnectViews.js';
import {Box} from 'ink';
import type {Command, CommandInput} from '../registry.js';

const NOTE_NOT_FOUND = -32_002;

/**
 * The note a command is about: an @mention, else the text as a note name, path
 * or alias (or a missing note's name, for /related), else the best fuzzy match.
 */
async function withNote<T>(input: CommandInput, api: VaultApi, call: (note: string) => Promise<T>): Promise<T | undefined> {
	const mention = input.mentions[0];
	if (mention) {
		return call(mention.path);
	}

	if (!input.text) {
		return undefined;
	}

	try {
		return await call(input.text);
	} catch (error) {
		if (!(error instanceof RpcError && error.code === NOTE_NOT_FOUND)) {
			throw error;
		}

		const best = (await api.pick({query: input.text, limit: 1})).results[0];
		if (!best) {
			throw new Error(`No note matches "${input.text}"`);
		}

		return call(best.path);
	}
}

export const related: Command = {
	name: 'related',
	aliases: ['r'],
	usage: '/related @note | /related <name>',
	summary: 'Notes related by links, text and tags, linked or not, each with a reason',
	async run(input, {api}) {
		const data = await withNote(input, api, async (note) => api.related({note, limit: 10}));
		if (!data) {
			throw new Error('Usage: /related @note (type @ to pick a note) or /related <name>');
		}

		return <RelatedView data={data} />;
	},
};

export const suggest: Command = {
	name: 'suggest',
	usage: '/suggest [links | notes | @note]',
	summary: 'Links to add between notes, and notes to write',
	async run(input, {api}) {
		const mode = input.text.toLowerCase();
		if (mode && mode !== 'links' && mode !== 'notes') {
			throw new Error('Usage: /suggest, /suggest links, /suggest notes or /suggest @note');
		}

		const mention = input.mentions[0];
		if (mention) {
			return <LinksView data={await api.suggestLinks({limit: 10, note: mention.path})} />;
		}

		if (mode === 'links') {
			return <LinksView data={await api.suggestLinks({limit: 10})} />;
		}

		if (mode === 'notes') {
			return <NotesToWriteView data={await api.suggestNotes({limit: 10})} />;
		}

		const [links, notes] = await Promise.all([api.suggestLinks({limit: 5}), api.suggestNotes({limit: 5})]);
		return (
			<Box flexDirection="column">
				<LinksView data={links} />
				<Box marginTop={1}>
					<NotesToWriteView data={notes} />
				</Box>
			</Box>
		);
	},
};

export const unlinked: Command = {
	name: 'unlinked',
	usage: '/unlinked [@note | <name>]',
	summary: 'Plain-text mentions of note titles that are not links yet',
	async run(input, {api}) {
		const data = (await withNote(input, api, async (note) => api.unlinked({note, limit: 10, per_note: 20}))) ?? (await api.unlinked({limit: 8, per_note: 3}));
		return <UnlinkedView data={data} />;
	},
};
