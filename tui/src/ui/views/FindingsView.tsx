import {Box, Text} from 'ink';
import type {FindingsListResult, VaultNextResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group, Headline} from './Section.js';

const METHODS: Array<[string, string]> = [
	['missing-note', 'a note many notes link to that does not exist'],
	['same-idea', 'two names for the same idea'],
	['existing-name', 'a link name that is probably an existing note'],
	['link', 'two notes that probably belong linked'],
];

function thresholds(data: FindingsListResult): string {
	const t = data.thresholds;
	const parts = [];
	if (t['missing-note']) {
		parts.push(`missing notes ${String(t['missing-note'].linking_notes)}+ linking notes`);
	}

	if (t['existing-name']) {
		parts.push(`names ${String(t['existing-name'].linking_notes)}+`);
	}

	if (t.link) {
		const cap = t.link.capped === true ? ` (the best ${String(t.link.limit)})` : '';
		parts.push(`links score ${Number(t.link.score).toFixed(2)}+${cap}`);
	}

	return parts.join(' · ');
}

/** findings.list: per method, each finding's stable id, title and evidence. */
export function FindingsView({data}: {data: FindingsListResult}) {
	const shown = thresholds(data);
	return (
		<Box flexDirection="column">
			<Headline title="Findings" detail={`${plural(data.total, 'finding')} · from the index${data.index.stale ? ' (out of date)' : ''}`} />
			<Body>
				{shown ? <Text dimColor>thresholds (percentiles of this vault, with floors): {shown}</Text> : null}
				<Text>
					<Text color={theme.accent}>/idea new {'<id>'}</Text> <Text dimColor>writes an idea from one (a dry run first; add write)</Text>
				</Text>
			</Body>
			{METHODS.map(([method, about]) => {
				const mine = data.findings.filter((finding) => finding.method === method);
				const total = data.by_method[method] ?? 0;
				if (total === 0) {
					return null;
				}

				return (
					<Group key={method} title={`${method}: ${about} (${formatNumber(total)})`}>
						{mine.map((finding) => (
							<Box key={finding.id} flexDirection="column">
								<Box>
									<Box flexShrink={0} marginRight={2}>
										<Text color={theme.accent}>{finding.id}</Text>
									</Box>
									<Box flexGrow={1} flexShrink={1}>
										<Text>{finding.title}</Text>
									</Box>
								</Box>
								{finding.evidence.map((line) => (
									<Box key={line} paddingLeft={4}>
										<Text dimColor>{line}</Text>
									</Box>
								))}
							</Box>
						))}
						{total > mine.length ? (
							<Text dimColor>
								… {formatNumber(total - mine.length)} more · <Text color={theme.accent}>/findings {method} all</Text>
							</Text>
						) : null}
					</Group>
				);
			})}
		</Box>
	);
}

/** The order `vault next` lists the kinds in. */
const KINDS = ['refresh', 'behind', 'merge', 'write', 'extract', 'page', 'idea'];

const ABOUT: Record<string, string> = {
	refresh: 'pages out of date with their zettels',
	behind: 'zettels a page is behind on',
	merge: 'two names for one idea',
	write: 'missing notes many notes link to',
	extract: 'sources no zettel links to',
	page: 'zettels in no page that share a tag',
	idea: 'open ideas nothing builds on',
};

/** vault.next: one queue, most trusted first; each item with why and the command that resolves it. */
export function NextView({data}: {data: VaultNextResult}) {
	const width = String(data.items.length).length;
	return (
		<Box flexDirection="column">
			<Headline title="Next" detail={`${plural(data.total, 'item')}, most trusted first`} />
			<Body>
				{data.items.length === 0 ? (
					<Text>
						<Text color={theme.ok}>✓</Text> nothing needs doing
					</Text>
				) : null}
				{data.items.map((item, index) => (
					<Box key={item.id} flexDirection="column">
						<Text>
							<Text dimColor>{String(index + 1).padStart(width)}.</Text> <Text color={theme.warn}>{item.kind.padEnd(8)}</Text>
							{item.title}
							<Text dimColor>   trust {item.trust}</Text>
						</Text>
						<Box paddingLeft={width + 2}>
							<Text dimColor>why {item.why}</Text>
						</Box>
						<Box paddingLeft={width + 2}>
							<Box flexShrink={0} marginRight={1}>
								<Text dimColor>run</Text>
							</Box>
							<Box flexGrow={1} flexShrink={1}>
								<Text color={theme.accent}>{item.command}</Text>
							</Box>
						</Box>
					</Box>
				))}
				{data.total > data.items.length ? (
					<Text dimColor>
						… {formatNumber(data.total - data.items.length)} more · <Text color={theme.accent}>/next all</Text>
					</Text>
				) : null}
			</Body>
			<Group title="By kind (listed / all)">
				{Object.entries(data.counts)
					.sort(([a], [b]) => KINDS.indexOf(a) - KINDS.indexOf(b))
					.map(([kind, count]) => (
						<Text key={kind}>
							{kind.padEnd(8)} {String(count.listed).padStart(4)} / {String(count.all).padEnd(4)}
							<Text dimColor>
								{' '}
								{ABOUT[kind] ?? ''}
								{count.zettels_in_no_page === null || count.zettels_in_no_page === undefined
									? ''
									: ` (${formatNumber(count.zettels_in_no_page)} zettels in no page)`}
								{count.all > count.listed ? '; below the median trust not listed' : ''}
							</Text>
						</Text>
					))}
			</Group>
		</Box>
	);
}
