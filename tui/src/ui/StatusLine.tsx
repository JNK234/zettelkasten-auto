import {Box, Text} from 'ink';
import type {IndexStatusResult} from '../rpc/schema.js';
import {formatNumber, theme} from './theme.js';

export type IndexState =
	| {phase: 'starting'}
	| {phase: 'indexing'}
	| {phase: 'ready'; status: IndexStatusResult}
	| {phase: 'error'; message: string};

function Freshness({index}: {index: IndexState}) {
	switch (index.phase) {
		case 'starting': {
			return <Text dimColor>◌ connecting…</Text>;
		}

		case 'indexing': {
			return <Text color={theme.warn}>◌ indexing…</Text>;
		}

		case 'error': {
			return <Text color={theme.error}>✗ {index.message}</Text>;
		}

		case 'ready': {
			const {status} = index;
			if (!status.built) {
				return <Text color={theme.error}>● index not built</Text>;
			}

			if (status.stale) {
				const pending = status.pending;
				const count = pending ? pending.added.length + pending.changed.length + pending.deleted.length : 0;
				const changed = count > 0 ? `${formatNumber(count)} file${count === 1 ? '' : 's'} changed` : undefined;
				const detail = pending?.reason ?? changed;
				return <Text color={theme.warn}>● index stale{detail ? ` (${detail})` : ''}</Text>;
			}

			return <Text color={theme.ok}>● index fresh</Text>;
		}
	}
}

export function StatusLine({
	vaultName,
	formatName,
	index,
	disconnected,
}: {
	vaultName: string;
	formatName?: string;
	index: IndexState;
	disconnected?: string;
}) {
	const status = index.phase === 'ready' ? index.status : undefined;
	return (
		<Box paddingX={1} justifyContent="space-between" gap={2}>
			{/* The vault facts never shrink; the key hint is truncated in narrow terminals. */}
			<Box flexShrink={0}>
				<Text>
				<Text color={theme.accent}>◆ {vaultName}</Text>
				{formatName ? <Text dimColor>  ·  format {formatName}</Text> : null}
				{status?.built ? (
					<Text dimColor>
						{'  ·  '}
						{formatNumber(status.notes)} notes · {formatNumber(status.links)} links
					</Text>
				) : null}
				<Text dimColor>{'  ·  '}</Text>
				{disconnected ? <Text color={theme.error}>✗ server disconnected</Text> : <Freshness index={index} />}
				</Text>
			</Box>
			<Box flexShrink={1}>
				<Text dimColor wrap="truncate">
					/help · @ note · ctrl+c quit
				</Text>
			</Box>
		</Box>
	);
}
