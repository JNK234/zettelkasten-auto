import {Box, Text} from 'ink';
import type {UncitedNote, VaultCoverageResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';

const REASONS: Array<{reason: UncitedNote['reason']; key: 'missing' | 'not_a_source' | 'none'; label: string}> = [
	{reason: 'missing', key: 'missing', label: 'cite only notes that do not exist'},
	{reason: 'not-a-source', key: 'not_a_source', label: 'cite notes that are not sources'},
	{reason: 'none', key: 'none', label: 'cite nothing at all'},
];

function Uncited({note}: {note: UncitedNote}) {
	const first = note.citations[0];
	return (
		<Text>
			<NotePath path={note.path} />
			<Text dimColor>  ({note.type})</Text>
			{first ? (
				<Text dimColor>
					{'  '}L{first.line} <Text color={note.reason === 'missing' ? theme.error : theme.warn}>{first.raw}</Text>
					{first.target_path ? ` → ${first.target_type ?? 'no type'}` : ''}
					{note.citations.length > 1 ? ` (+${note.citations.length - 1})` : ''}
				</Text>
			) : null}
		</Text>
	);
}

export function CoverageView({data, perGroup}: {data: VaultCoverageResult; perGroup?: number}) {
	const {sources, citing} = data;
	const unlinkedTotal = sources.total - sources.linked;
	return (
		<Box flexDirection="column">
			<Headline title="Coverage" detail={`${data.format.name} · sources: ${data.source_types.join(', ')} · citing: ${data.citing_types.join(', ')}`} />
			<Body>
				<Text>
					<Text color={unlinkedTotal > 0 ? theme.warn : theme.ok}>{unlinkedTotal > 0 ? '!' : '✓'}</Text> {plural(sources.total, 'source')}:{' '}
					{formatNumber(sources.linked)} linked, {formatNumber(unlinkedTotal)} that nothing links to
				</Text>
				<Text>
					<Text color={citing.without_source > 0 ? theme.warn : theme.ok}>{citing.without_source > 0 ? '!' : '✓'}</Text>{' '}
					{plural(citing.total, 'citing note')}: {formatNumber(citing.cite_a_source)} cite a source, {formatNumber(citing.without_source)} cite none
				</Text>
				{citing.without_source > 0 ? (
					// Indent with padding, not leading spaces: spaces inside wrapped text push it past the edge.
					<Box paddingLeft={2}>
						<Text dimColor>{REASONS.map(({key, label}) => `${formatNumber(citing.by_reason[key])} ${label}`).join(' · ')}</Text>
					</Box>
				) : null}
			</Body>
			{sources.unlinked.length > 0 ? (
				<Group title={`Sources nothing links to (${formatNumber(unlinkedTotal)})`}>
					{sources.unlinked.slice(0, perGroup).map((path) => (
						<NotePath key={path} path={path} />
					))}
					{unlinkedTotal > Math.min(sources.unlinked.length, perGroup ?? Infinity) ? (
						<Text dimColor>… {formatNumber(unlinkedTotal - Math.min(sources.unlinked.length, perGroup ?? Infinity))} more</Text>
					) : null}
				</Group>
			) : null}
			{REASONS.map(({reason, key, label}) => {
				const notes = citing.notes.filter((note) => note.reason === reason).slice(0, perGroup);
				const total = citing.by_reason[key];
				if (total === 0) {
					return null;
				}

				return (
					<Group key={reason} title={`Notes that ${label} (${formatNumber(total)})`}>
						{notes.map((note) => (
							<Uncited key={note.path} note={note} />
						))}
						{total > notes.length ? <Text dimColor>… {formatNumber(total - notes.length)} more · /coverage all lists every note</Text> : null}
					</Group>
				);
			})}
			{data.most_cited.length > 0 ? (
				<Group title="Most cited sources (citing notes)">
					{data.most_cited.map((row) => (
						<Text key={row.path}>
							<Text>{String(row.citing_notes).padStart(4)}  </Text>
							<NotePath path={row.path} />
						</Text>
					))}
				</Group>
			) : null}
		</Box>
	);
}
