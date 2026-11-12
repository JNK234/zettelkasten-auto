import {Box, Text} from 'ink';
import type {VaultPipelineResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';

const REASONS: Record<string, string> = {
	missing: 'cite sources not in the vault',
	'not-a-source': 'cite notes that are not sources',
	'url-only': 'cite only URLs',
	none: 'cite nothing',
};

function More({shown, total, hint}: {shown: number; total: number; hint: string}) {
	return total > shown ? (
		<Text dimColor>
			… {formatNumber(total - shown)} more · <Text color={theme.accent}>{hint}</Text>
		</Text>
	) : null;
}

/** vault.pipeline: each source's stage, the counts, and the gaps (sources not extracted, zettels in no page). */
export function PipelineView({data}: {data: VaultPipelineResult}) {
	const role = (key: 'sources' | 'extract' | 'compile') => data.roles[key].join('/');
	const [source, zettel, page] = [role('sources'), role('extract'), role('compile')];
	const total = data.stages.ingested + data.stages.extracted + data.stages.compiled;
	const {unsourced} = data.extract;
	const reasons = Object.entries(unsourced.by_reason)
		.filter(([, count]) => count > 0)
		.map(([reason, count]) => `${formatNumber(count)} ${REASONS[reason] ?? reason}`)
		.join(', ');
	const meaning = {
		compiled: `a ${page} links to a ${zettel} that links to it`,
		extracted: `a ${zettel} links to it, no ${page} uses those yet`,
		ingested: `no ${zettel} links to it`,
	};
	const limit = data.limit ?? Number.POSITIVE_INFINITY;
	const stagesTotal = (stage: 'compiled' | 'extracted' | 'ingested') => data.stages[stage];
	return (
		<Box flexDirection="column">
			<Headline title="Pipeline" detail={`${data.format.name} · ${plural(total, `${source} note`)}`} />
			<Body>
				{(['compiled', 'extracted', 'ingested'] as const).map((stage) => (
					<Text key={stage}>
						<Text color={stage === 'ingested' ? theme.warn : theme.ok}>{stage.padEnd(10)}</Text>
						{String(stagesTotal(stage)).padStart(5)}
						<Text dimColor>   {meaning[stage]}</Text>
					</Text>
				))}
				<Text>
					{plural(data.extract.total, `${zettel} note`)}: {formatNumber(data.extract.in_a_compile_note)} in a {page} ·{' '}
					{formatNumber(data.extract.resting_on_a_source)} resting on a {source}
					{unsourced.total > 0 ? (
						<Text color={theme.warn}>
							{' '}
							· {formatNumber(unsourced.total)} on no {source}
						</Text>
					) : null}
				</Text>
				{reasons ? <Text dimColor>on no {source}: {reasons}</Text> : null}
				<Text>
					{plural(data.compile.total, `${page} note`)}: {formatNumber(data.compile.using_an_extract_note)} link to a {zettel}
					{data.ignored > 0 ? <Text dimColor> · {plural(data.ignored, 'note')} in ignored folders</Text> : null}
				</Text>
			</Body>
			{data.sources.length > 0 ? (
				<Group title={`${source[0]!.toUpperCase()}${source.slice(1)} notes by stage (${zettel}s linking to it, ${page}s using those)`}>
					{data.sources.map((row) => (
						<Box key={row.path}>
							<Box width={11} flexShrink={0}>
								<Text color={row.stage === 'ingested' ? theme.warn : theme.ok}>{row.stage}</Text>
							</Box>
							<Box width={12} flexShrink={0}>
								<Text dimColor>
									{String(row.zettels.length).padStart(3)} {zettel[0]} {String(row.pages.length).padStart(3)} {page[0]}
								</Text>
							</Box>
							<NotePath path={row.path} />
						</Box>
					))}
					<More shown={data.sources.length} total={total} hint="/pipeline all" />
				</Group>
			) : null}
			<Group title={`Not extracted yet (${formatNumber(data.stages.ingested)})`}>
				{data.gaps.not_extracted.map((path) => (
					<NotePath key={path} path={path} />
				))}
				{limit < Number.POSITIVE_INFINITY ? <More shown={data.gaps.not_extracted.length} total={data.stages.ingested} hint="/pipeline all" /> : null}
			</Group>
			<Group title={`${zettel[0]!.toUpperCase()}${zettel.slice(1)}s in no ${page} (${formatNumber(data.extract.total - data.extract.in_a_compile_note)})`}>
				{data.gaps.not_compiled.map((path) => (
					<NotePath key={path} path={path} />
				))}
				<More shown={data.gaps.not_compiled.length} total={data.extract.total - data.extract.in_a_compile_note} hint="/pipeline all" />
			</Group>
		</Box>
	);
}
