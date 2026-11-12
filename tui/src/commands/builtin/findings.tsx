import type {FindingsListParams} from '../../rpc/schema.js';
import {FindingsView} from '../../ui/views/FindingsView.js';
import {parseLimit, takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const USAGE = '/findings [missing-note | same-idea | existing-name | link]… [all | <per method>] [<id>]';
type Method = NonNullable<FindingsListParams['methods']>[number];
const METHODS = new Set<string>(['missing-note', 'same-idea', 'existing-name', 'link']);
const ID = /^[a-z]+-[\da-f]{10}$/;

export const findings: Command = {
	name: 'findings',
	usage: USAGE,
	summary: 'What the engine discovered (missing notes, two names for one idea, links to add), each with a stable id',
	async run(input, {api}) {
		const tokens = tokenize(input.text);
		const all = takeFlag(tokens, 'all');
		const methods: Method[] = [];
		let id: string | undefined;
		let limit: number | undefined = all ? undefined : 3;
		for (const token of tokens) {
			if (METHODS.has(token.value)) {
				methods.push(token.value as Method);
			} else if (ID.test(token.value)) {
				id = token.value;
			} else if (/^\d+$/.test(token.value)) {
				limit = parseLimit(token.value, USAGE);
			} else {
				throw new Error(`Usage: ${USAGE} (did not understand "${token.value}")`);
			}
		}

		const params: FindingsListParams = {limit: id ? null : (limit ?? null)};
		if (methods.length > 0) {
			params.methods = methods as [Method, ...Method[]];
		}

		if (id) {
			params.id = id;
		}

		const data = await withFormat(async () => api.findings(params));
		return <FindingsView data={data} />;
	},
};
