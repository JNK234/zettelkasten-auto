import {PassThrough} from 'node:stream';
import {describe, expect, test} from 'vitest';
import {activeMention, formatMention, parseMentions} from '../src/commands/mentions.js';
import {type Command, CommandRegistry} from '../src/commands/registry.js';
import {createVaultApi} from '../src/rpc/api.js';
import {ConnectionClosedError, JsonRpcClient, RpcError} from '../src/rpc/client.js';
import {PromptHistory} from '../src/state/history.js';

const command = (name: string, aliases?: string[]): Command => ({name, aliases, summary: `${name} summary`, run: async () => name});

describe('CommandRegistry', () => {
	test('parses commands, args and mentions', () => {
		const registry = new CommandRegistry().register(command('note', ['n']));
		const parsed = registry.parse('/n @"Notes/Reading notes.md"  extra words ');
		expect(parsed.kind).toBe('command');
		if (parsed.kind === 'command') {
			expect(parsed.command.name).toBe('note');
			expect(parsed.input.mentions.map((m) => m.path)).toEqual(['Notes/Reading notes.md']);
			expect(parsed.input.text).toBe('extra words');
		}
	});

	test('empty, plain text and unknown commands', () => {
		const registry = new CommandRegistry().register(command('stats'), command('health'), command('help'));
		expect(registry.parse('   ')).toEqual({kind: 'empty'});
		expect(registry.parse('hello')).toEqual({kind: 'text', text: 'hello'});
		const unknown = registry.parse('/hlep');
		expect(unknown.kind).toBe('unknown');
		if (unknown.kind === 'unknown') {
			expect(unknown.suggestions.map((c) => c.name)).toEqual(['health', 'help']);
		}
	});

	test('suggest by prefix, including aliases; list is sorted', () => {
		const registry = new CommandRegistry().register(command('stats'), command('note', ['n']), command('help', ['h']));
		expect(registry.suggest('').map((c) => c.name)).toEqual(['help', 'note', 'stats']);
		expect(registry.suggest('n').map((c) => c.name)).toEqual(['note']);
		expect(registry.suggest('x')).toEqual([]);
		expect(registry.get('N')?.name).toBe('note');
	});

	test('rejects duplicate and invalid names', () => {
		const registry = new CommandRegistry().register(command('note', ['n']));
		expect(() => registry.register(command('n'))).toThrow('/n is already registered');
		expect(() => registry.register(command('Bad Name'))).toThrow('Invalid command name');
	});
});

describe('mentions', () => {
	test('parse quoted and bare mentions, ignoring emails', () => {
		const {mentions, rest} = parseMentions('compare @alpha.md with @"Folder/Beta note.md" mail a@b.c');
		expect(mentions.map((m) => m.path)).toEqual(['alpha.md', 'Folder/Beta note.md']);
		expect(rest).toBe('compare with mail a@b.c');
	});

	test('format quotes only when needed', () => {
		expect(formatMention('a/b.md')).toBe('@a/b.md');
		expect(formatMention('a/b c.md')).toBe('@"a/b c.md"');
	});

	test('active mention is the @word at the cursor', () => {
		expect(activeMention('/note @kl', 9)).toEqual({start: 6, query: 'kl'});
		expect(activeMention('/note @', 7)).toEqual({start: 6, query: ''});
		expect(activeMention('/note @kl pen', 13)).toBeUndefined(); // a space ends the mention
		expect(activeMention('/note @"A b.md" ', 16)).toBeUndefined(); // completed, quoted
		expect(activeMention('mail a@b', 8)).toBeUndefined(); // not after whitespace
		expect(activeMention('/note @kl', 7)).toEqual({start: 6, query: ''}); // cursor inside
	});
});

describe('PromptHistory', () => {
	test('up/down with the draft restored', () => {
		const history = new PromptHistory();
		history.add('/stats');
		history.add('/health');
		history.add('/health'); // consecutive duplicate stored once
		expect(history.size).toBe(2);
		expect(history.previous('/no')).toBe('/health');
		expect(history.previous('/health')).toBe('/stats');
		expect(history.previous('/stats')).toBeUndefined();
		expect(history.next()).toBe('/health');
		expect(history.next()).toBe('/no');
		expect(history.next()).toBeUndefined();
	});

	test('limit drops the oldest', () => {
		const history = new PromptHistory(2);
		for (const line of ['a', 'b', 'c']) history.add(line);
		expect(history.previous('')).toBe('c');
		expect(history.previous('')).toBe('b');
		expect(history.previous('')).toBeUndefined();
	});
});

describe('JsonRpcClient', () => {
	function connect() {
		const toServer = new PassThrough();
		const fromServer = new PassThrough();
		const client = new JsonRpcClient(fromServer, toServer);
		const sent: any[] = [];
		toServer.on('data', (chunk: Buffer) => sent.push(...chunk.toString().split('\n').filter(Boolean).map((l) => JSON.parse(l))));
		const reply = (message: unknown) => fromServer.write(`${JSON.stringify(message)}\n`);
		return {client, sent, reply, fromServer};
	}

	test('one line per request; responses matched by id, in any order', async () => {
		const {client, sent, reply} = connect();
		const first = client.request('a', {x: 1});
		const second = client.request('b');
		await new Promise((r) => setImmediate(r));
		expect(sent).toEqual([
			{jsonrpc: '2.0', id: 1, method: 'a', params: {x: 1}},
			{jsonrpc: '2.0', id: 2, method: 'b'},
		]);
		reply({jsonrpc: '2.0', id: 2, result: 'two'});
		reply({jsonrpc: '2.0', id: 1, result: 'one'});
		await expect(first).resolves.toBe('one');
		await expect(second).resolves.toBe('two');
	});

	test('error responses become RpcError with code and data', async () => {
		const {client, reply} = connect();
		const pending = client.request('note.get', {path: 'x'});
		reply({jsonrpc: '2.0', id: 1, error: {code: -32_002, message: 'Note not found', data: {path: 'x'}}});
		const error = await pending.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(RpcError);
		expect(error).toMatchObject({code: -32_002, data: {path: 'x'}});
	});

	test('closing the stream rejects pending and later requests', async () => {
		const {client, fromServer} = connect();
		const pending = client.request('slow');
		fromServer.end();
		await expect(pending).rejects.toBeInstanceOf(ConnectionClosedError);
		await expect(client.request('later')).rejects.toBeInstanceOf(ConnectionClosedError);
		expect(client.isClosed).toBe(true);
	});

	test('unparseable or unmatched lines are reported, not fatal', async () => {
		const {client, reply, fromServer} = connect();
		const problems: string[] = [];
		client.events.on('protocol-error', (line) => problems.push(line));
		fromServer.write('not json\n');
		reply({jsonrpc: '2.0', id: 99, result: null});
		const ok = client.request('a');
		reply({jsonrpc: '2.0', id: 1, result: 'fine'});
		await expect(ok).resolves.toBe('fine');
		expect(problems).toHaveLength(2);
	});

	test('VaultApi maps to the engine method names', async () => {
		const calls: Array<[string, unknown]> = [];
		const api = createVaultApi({request: async (method, params) => (calls.push([method, params]), undefined as never)});
		await api.build({full: true});
		await api.status();
		await api.stats({top: 5});
		await api.health({limit: 3});
		await api.pick({query: 'q', limit: 2});
		await api.note('a.md');
		expect(calls).toEqual([
			['index.build', {full: true}],
			['index.status', undefined],
			['vault.stats', {top: 5}],
			['health.report', {limit: 3}],
			['notes.pick', {query: 'q', limit: 2}],
			['note.get', {path: 'a.md'}],
		]);
	});
});
