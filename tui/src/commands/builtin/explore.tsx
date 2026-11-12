import {RpcError} from '../../rpc/client.js';
import type {ExploreRunResult} from '../../rpc/schema.js';
import {ExploreBrowser, type ExploreStart} from '../../ui/views/ExploreView.js';
import {ConfigView, KeptView, MapSinceView, MapView} from '../../ui/views/WorkbenchViews.js';
import {splitPath} from '../../ui/theme.js';
import {takeFlag, tokenize} from '../args.js';
import {type Command, interactive} from '../registry.js';

const NOTE_NOT_FOUND = -32_002;
const USAGE = '/explore [@note | <name>] [--fresh]';

export const explore: Command = {
	name: 'explore',
	aliases: ['x'],
	usage: USAGE,
	summary: 'Explore from a note (or where to start): a section per lens; k keep, d dismiss, l link it, p pivot',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const fresh = takeFlag(tokens, '--fresh', 'fresh');
		const name = tokens.map((token) => token.value).join(' ');
		const mention = input.mentions[0];
		let start: ExploreStart = {label: 'where to start'};
		if (mention) {
			start = {note: mention.path, label: splitPath(mention.path).title};
		} else if (name) {
			start = {note: name, label: name};
		}

		const call = async (note: string | undefined): Promise<ExploreRunResult> =>
			api.explore({...(note === undefined ? {} : {note}), ...(fresh ? {fresh: true} : {})});
		let data: ExploreRunResult;
		try {
			data = await call(start.note);
		} catch (error) {
			if (!(error instanceof RpcError && error.code === NOTE_NOT_FOUND && name && !mention)) {
				throw error;
			}

			const best = (await api.pick({query: name, limit: 1})).results[0];
			if (!best) {
				throw new Error(`No note matches "${name}"`);
			}

			start = {note: best.path, label: splitPath(best.path).title};
			data = await call(start.note);
		}

		return interactive((done) => <ExploreBrowser api={api} data={data} start={start} done={done} />);
	},
};

export const kept: Command = {
	name: 'kept',
	usage: '/kept [lens]',
	summary: 'What you kept from /explore, with its lens and evidence',
	async run(input, {api}) {
		const lens = input.text.trim();
		return <KeptView data={await api.kept(lens ? {lens: lens as never} : {})} />;
	},
};

const MAP_LIMIT = 5;

export const map: Command = {
	name: 'map',
	usage: '/map [record] | /map since <date>',
	summary: 'The map: communities, topic pages, bridges; record saves a snapshot, since <date> says what changed',
	async run(input, {api}) {
		const words = input.text.trim().split(/\s+/).filter(Boolean);
		if (words[0] === 'since') {
			if (!words[1]) {
				throw new Error('Usage: /map since <date>, e.g. /map since 2026-11-10');
			}

			return <MapSinceView data={await api.mapSince({since: words[1]})} />;
		}

		if (words.length > 0 && words[0] !== 'record') {
			throw new Error('Usage: /map, /map record or /map since <date>');
		}

		const record = words[0] === 'record';
		return <MapView data={await api.map({limit: MAP_LIMIT, ...(record ? {record: true} : {})})} limit={MAP_LIMIT} />;
	},
};

export const config: Command = {
	name: 'config',
	summary: 'Every engine setting, its value and where it came from',
	async run(_input, {api}) {
		return <ConfigView data={await api.config()} />;
	},
};
