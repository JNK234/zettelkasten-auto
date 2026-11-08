import {RpcError} from '../../rpc/client.js';
import {QueryErrorView, SearchBrowser, SearchSummary} from '../../ui/views/SearchView.js';
import {type Command, interactive} from '../registry.js';

export const INVALID_QUERY = -32_006;
export const SEARCH_LIMIT = 30;

export const search: Command = {
	name: 'search',
	aliases: ['s'],
	usage: '/search <query>',
	summary: 'Search text and structure (tag:, path:, links-to:, "phrases", NEAR/n…); ↑↓ Enter o Esc',
	async run(input, {api, openNote}) {
		const query = input.args.trim();
		if (!query) {
			throw new Error('Usage: /search <query>, e.g. /search reward hacking tag:ai-safety -path:10_SOURCES');
		}

		let result;
		try {
			result = await api.search({query, limit: SEARCH_LIMIT});
		} catch (error) {
			if (error instanceof RpcError && error.code === INVALID_QUERY) {
				const data = (error.data ?? {}) as {start?: number | null; end?: number | null; hint?: string | null};
				return <QueryErrorView query={query} message={error.message} start={data.start} end={data.end} hint={data.hint} />;
			}

			throw error;
		}

		if (result.hits.length === 0) {
			return <SearchSummary result={result} />;
		}

		return interactive((done) => <SearchBrowser result={result} api={api} openNote={openNote} done={done} />);
	},
};
