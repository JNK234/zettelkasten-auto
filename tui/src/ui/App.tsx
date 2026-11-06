/**
 * Layout, top to bottom (like Claude Code):
 *   transcript   - <Static>: each finished command is printed once and then
 *                  scrolls with the terminal, never re-rendered
 *   activity     - spinner while a command runs
 *   prompt       - input box, with the "/" or "@" list under it
 *   status line  - vault, counts, index freshness
 */
import {basename, resolve} from 'node:path';
import {Box, Static, Text, useAnimation, useApp} from 'ink';
import {type ReactNode, useCallback, useEffect, useMemo, useRef, useState} from 'react';
import type {CommandRegistry} from '../commands/registry.js';
import {ConnectionClosedError, RpcError} from '../rpc/client.js';
import type {VaultApi} from '../rpc/api.js';
import {PromptHistory} from '../state/history.js';
import {Prompt} from './Prompt.js';
import {type IndexState, StatusLine} from './StatusLine.js';
import {theme} from './theme.js';

type TranscriptItem = {kind: 'output'; line: string; output: ReactNode} | {kind: 'error'; line: string; message: string};
type Entry = {id: number; kind: 'banner'} | (TranscriptItem & {id: number});

export type AppProps = {
	api: VaultApi;
	registry: CommandRegistry;
	vaultPath: string;
	/** Run an incremental index.build on start (default true). */
	autoIndex?: boolean;
	/** Subscribe to connection loss; returns an unsubscribe function. */
	onDisconnect?: (listener: (error: Error) => void) => () => void;
};

export function describeError(error: unknown): string {
	if (error instanceof RpcError) {
		return `${error.message} (error ${error.code})`;
	}

	if (error instanceof ConnectionClosedError) {
		return error.message;
	}

	return error instanceof Error ? error.message : String(error);
}

function Banner({vaultPath}: {vaultPath: string}) {
	return (
		<Box flexDirection="column" borderStyle="round" borderColor={theme.accent} paddingX={1} marginBottom={1}>
			<Text>
				<Text color={theme.accent}>✻ </Text>
				<Text bold>vault-tui</Text>
				<Text dimColor>  a terminal workbench for your Obsidian vault</Text>
			</Text>
			<Text> </Text>
			<Text dimColor>  /help for commands · @ to pick a note · ↑↓ history</Text>
			<Text dimColor>  vault: {vaultPath}</Text>
		</Box>
	);
}

function Echo({line}: {line: string}) {
	return (
		<Text dimColor>
			{'> '}
			{line}
		</Text>
	);
}

function TranscriptEntry({entry}: {entry: TranscriptItem}) {
	return (
		<Box flexDirection="column" marginBottom={1}>
			<Echo line={entry.line} />
			{entry.kind === 'error' ? (
				<Text>
					<Text color={theme.error}>⏺ </Text>
					<Text color={theme.error}>{entry.message}</Text>
				</Text>
			) : (
				entry.output
			)}
		</Box>
	);
}

const SPINNER = ['✶', '✸', '✹', '✺', '✹', '✷'];

function Activity({line}: {line: string}) {
	const {frame} = useAnimation({interval: 120});
	return (
		<Box flexDirection="column">
			<Echo line={line} />
			<Text>
				<Text color={theme.accent}>{SPINNER[frame % SPINNER.length]} </Text>
				<Text dimColor>Working…</Text>
			</Text>
		</Box>
	);
}

export function App({api, registry, vaultPath, autoIndex = true, onDisconnect}: AppProps) {
	const {exit} = useApp();
	const history = useMemo(() => new PromptHistory(), []);
	const nextId = useRef(1);
	const [entries, setEntries] = useState<Entry[]>([{id: 0, kind: 'banner'}]);
	const [running, setRunning] = useState<string | undefined>();
	const [index, setIndex] = useState<IndexState>({phase: autoIndex ? 'indexing' : 'starting'});
	const [disconnected, setDisconnected] = useState<string | undefined>();

	const refreshStatus = useCallback(async () => {
		try {
			setIndex({phase: 'ready', status: await api.status()});
		} catch (error) {
			setIndex({phase: 'error', message: describeError(error)});
		}
	}, [api]);

	useEffect(() => {
		void (async () => {
			if (autoIndex) {
				try {
					await api.build();
				} catch (error) {
					setIndex({phase: 'error', message: describeError(error)});
					return;
				}
			}

			await refreshStatus();
		})();
	}, [api, autoIndex, refreshStatus]);

	useEffect(
		() =>
			onDisconnect?.((error) => {
				setDisconnected(error.message);
			}),
		[onDisconnect],
	);

	const push = useCallback((item: TranscriptItem) => {
		setEntries((list) => [...list, {...item, id: nextId.current++}]);
	}, []);

	const run = useCallback(
		async (line: string) => {
			const parsed = registry.parse(line);
			switch (parsed.kind) {
				case 'empty': {
					return;
				}

				case 'text': {
					push({kind: 'error', line, message: 'Not a command. Commands start with "/"; try /help.'});
					return;
				}

				case 'unknown': {
					const hint = parsed.suggestions.length > 0 ? `Did you mean ${parsed.suggestions.map((c) => `/${c.name}`).join(', ')}?` : 'Try /help.';
					push({kind: 'error', line, message: `Unknown command /${parsed.name}. ${hint}`});
					return;
				}

				case 'command': {
					setRunning(line);
					try {
						const output = await parsed.command.run(parsed.input, {api, registry});
						push({kind: 'output', line, output});
					} catch (error) {
						push({kind: 'error', line, message: describeError(error)});
					} finally {
						setRunning(undefined);
					}

					void refreshStatus();
				}
			}
		},
		[api, registry, push, refreshStatus],
	);

	return (
		<Box flexDirection="column">
			<Static items={entries}>
				{(entry) => (entry.kind === 'banner' ? <Banner key={entry.id} vaultPath={vaultPath} /> : <TranscriptEntry key={entry.id} entry={entry} />)}
			</Static>
			{running ? <Activity line={running} /> : null}
			{disconnected ? (
				<Text color={theme.error}>⏺ {disconnected}. Restart vault-tui to reconnect.</Text>
			) : null}
			<Prompt
				api={api}
				registry={registry}
				history={history}
				busy={running !== undefined}
				onSubmit={(line) => {
					void run(line);
				}}
				onExit={exit}
			/>
			<StatusLine vaultName={basename(resolve(vaultPath))} index={index} disconnected={disconnected} />
		</Box>
	);
}
