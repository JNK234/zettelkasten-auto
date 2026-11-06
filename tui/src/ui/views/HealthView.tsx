import {Box, Text} from 'ink';
import type {HealthReportResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';

type Severity = 'error' | 'warn';

function Check({count, label, severity}: {count: number; label: string; severity: Severity}) {
	if (count === 0) {
		return (
			<Text>
				<Text color={theme.ok}>✓</Text> {formatNumber(count)} {label}
			</Text>
		);
	}

	return (
		<Text>
			<Text color={severity === 'error' ? theme.error : theme.warn}>{severity === 'error' ? '✗' : '!'}</Text> {formatNumber(count)}{' '}
			{label}
		</Text>
	);
}

function shownOf(total: number, listed: number): string {
	return listed < total ? ` (showing ${listed} of ${formatNumber(total)})` : '';
}

function PathGroup({title, paths, total}: {title: string; paths: string[]; total: number}) {
	if (paths.length === 0) {
		return null;
	}

	return (
		<Group title={`${title}${shownOf(total, paths.length)}`}>
			{paths.map((path) => (
				<NotePath key={path} path={path} />
			))}
		</Group>
	);
}

export function HealthView({data}: {data: HealthReportResult}) {
	const s = data.summary;
	const shown = (total: number, list: unknown[]) => shownOf(total, list.length);
	return (
		<Box flexDirection="column">
			<Headline title="Vault health" />
			<Body>
				<Check count={s.unresolved_links} label={`unresolved links (${plural(s.unresolved_targets, 'missing target')})`} severity="error" />
				<Check count={s.ambiguous_links} label="ambiguous links" severity="error" />
				<Check count={s.missing_subpaths} label="missing heading/block targets" severity="error" />
				<Check count={s.orphans} label="orphans (no backlinks)" severity="warn" />
				<Check count={s.dead_ends} label="dead ends (no outgoing note links)" severity="warn" />
				<Check count={s.duplicate_basenames} label="duplicate basenames" severity="warn" />
				<Check count={s.empty_notes} label="empty notes" severity="warn" />
			</Body>
			{data.unresolved_links.length > 0 ? (
				<Group title={`Unresolved targets${shown(s.unresolved_targets, data.unresolved_links)}`}>
					{data.unresolved_links.map((group) => {
						const sources = new Set(group.links.map((link) => link.source)).size;
						return (
							<Text key={group.target}>
								<Text color={theme.error}>{group.target}</Text>
								<Text dimColor>  {plural(group.links.length, 'link')} from {plural(sources, 'note')}</Text>
							</Text>
						);
					})}
				</Group>
			) : null}
			{data.ambiguous_links.length > 0 ? (
				<Group title={`Ambiguous links${shown(s.ambiguous_links, data.ambiguous_links)}`}>
					{data.ambiguous_links.map((link) => (
						<Text key={`${link.source}:${link.line}:${link.raw}`}>
							{link.raw}
							<Text dimColor>  {link.source}:{link.line} → {link.candidates.join(' | ')}</Text>
						</Text>
					))}
				</Group>
			) : null}
			{data.missing_subpaths.length > 0 ? (
				<Group title={`Missing heading/block targets${shown(s.missing_subpaths, data.missing_subpaths)}`}>
					{data.missing_subpaths.map((link) => (
						<Text key={`${link.source}:${link.line}:${link.raw}`}>
							{link.raw}
							<Text dimColor>
								{'  '}
								{link.kind} not found · {link.source}:{link.line}
							</Text>
						</Text>
					))}
				</Group>
			) : null}
			<PathGroup title="Orphans" paths={data.orphans} total={s.orphans} />
			<PathGroup title="Dead ends" paths={data.dead_ends} total={s.dead_ends} />
			<PathGroup title="Empty notes" paths={data.empty_notes} total={s.empty_notes} />
			{data.duplicate_basenames.length > 0 ? (
				<Group title="Duplicate basenames">
					{data.duplicate_basenames.map((duplicate) => (
						<Text key={duplicate.name}>
							{duplicate.name}
							<Text dimColor>  {duplicate.paths.join(' | ')}</Text>
						</Text>
					))}
				</Group>
			) : null}
		</Box>
	);
}
