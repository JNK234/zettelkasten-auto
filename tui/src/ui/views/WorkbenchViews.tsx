/**
 * /kept, /map (and /map since), /config: what the engine remembers, the map and its history, the settings.
 */
import {Box, Text} from 'ink';
import type {ConfigShowResult, ExploreKeptResult, MapResult, MapSinceResult} from '../../rpc/schema.js';
import {formatNumber, plural, splitPath, theme} from '../theme.js';
import {Body, Group, Headline} from './Section.js';

const title = (path: string) => splitPath(path).title;

export function KeptView({data}: {data: ExploreKeptResult}) {
	return (
		<Box flexDirection="column">
			<Headline title="Kept" detail={`${plural(data.kept.length, 'result')} · ${data.memory}`} />
			{data.kept.length === 0 ? (
				<Body>
					<Text dimColor>nothing kept yet: k in /explore keeps the selected result</Text>
				</Body>
			) : (
				<Box flexDirection="column" paddingLeft={5}>
					{data.kept.map((item) => (
						<Box key={`${item.start ?? ''} ${item.lens} ${item.target}`} flexDirection="column" marginTop={1}>
							<Text wrap="truncate-end">
								<Text bold>{item.title}</Text>
								<Text dimColor>  from {item.start ? title(item.start) : 'no starting point'} · {item.at.slice(0, 19)}</Text>
							</Text>
							<Text dimColor wrap="truncate-end">
								{'  '}
								{item.lens} · {item.evidence}
							</Text>
						</Box>
					))}
				</Box>
			)}
		</Box>
	);
}

export function MapView({data, limit}: {data: MapResult; limit: number}) {
	const bare = data.communities.filter((community) => community.topic_pages.length === 0);
	return (
		<Box flexDirection="column">
			<Headline
				title={`Map · ${plural(data.communities.length, 'community', 'communities')}`}
				detail={`modularity ${data.modularity.toFixed(2)} · ${formatNumber(data.notes_in_communities)} of ${formatNumber(data.notes)} notes in them · ${plural(data.alone, 'note')} in none`}
			/>
			<Body>
				<Text dimColor>topic pages: {data.topic_pages}</Text>
				{data.recorded ? (
					<Text color={theme.ok}>
						✓ recorded a snapshot ({formatNumber(data.recorded.bytes)} bytes, {data.recorded.gaps} gaps): /map since {data.recorded.at.slice(0, 10)} compares
					</Text>
				) : null}
			</Body>
			<Group title="Communities">
				{data.communities.slice(0, limit).map((community) => (
					<Text key={community.number} wrap="truncate-end">
						<Text bold>{`${community.number}. ${community.name}`}</Text>
						<Text dimColor>
							{'  '}
							{community.notes.length} notes · hubs {community.hubs.map(title).join(', ')}
							{community.topic_pages.length > 0 ? ` · topic page ${title(community.topic_pages[0]!.path)}` : ''}
						</Text>
					</Text>
				))}
				{data.communities.length > limit ? <Text dimColor>… {data.communities.length - limit} more</Text> : null}
			</Group>
			{bare.length > 0 && data.topic_pages.startsWith('notes the format') ? (
				<Group title={`No topic page (${bare.length} of ${data.communities.length})`}>
					<Text dimColor wrap="truncate-end">
						{bare.slice(0, limit).map((community) => `${community.number}. ${community.name}`).join(' · ')}
					</Text>
				</Group>
			) : null}
			<Group title="Bridge notes">
				{data.bridges.slice(0, Math.min(limit, 5)).map((bridge) => (
					<Text key={bridge.path} wrap="truncate-end">
						{title(bridge.path)}
						<Text dimColor>
							{'  '}
							betweenness {bridge.betweenness.toFixed(3)} (p{Math.round(bridge.betweenness_percentile)}) · reaches {bridge.spans.slice(0, 5).join(', ')}
						</Text>
					</Text>
				))}
			</Group>
		</Box>
	);
}

export function MapSinceView({data}: {data: MapSinceResult}) {
	const closed = data.gaps.filter((gap) => gap.status === 'closed').length;
	return (
		<Box flexDirection="column">
			<Headline title={`Since ${data.snapshot}`} detail={data.chosen} />
			<Body>
				<Text dimColor>
					notes {data.notes[0]} → {data.notes[1]} · links {data.links[0]} → {data.links[1]} · communities {data.communities[0]} →{' '}
					{data.communities[1]}
				</Text>
			</Body>
			<Group title={`Gaps: ${closed} of ${data.gaps.length} closed`}>
				{data.gaps.map((gap) => (
					<Box key={gap.communities.join('-')} flexDirection="column">
						<Text wrap="truncate-end">
							<Text color={gap.status === 'closed' ? theme.ok : undefined}>{gap.status.padEnd(7)}</Text> {gap.names.join(' / ')}
						</Text>
						<Text dimColor wrap="truncate-end">
							{'        '}
							{gap.why}
						</Text>
						{gap.added.map((pair) => (
							<Text key={pair.join()} dimColor wrap="truncate-end">
								{'        '}new link {pair.map(title).join(' ↔ ')}
							</Text>
						))}
					</Box>
				))}
			</Group>
			<Group title={`Communities: formed ${data.formed.length}, gone ${data.gone.length}`}>
				{[...data.formed.map((c) => `formed ${c.community}. ${c.name} (${c.notes} notes)`), ...data.gone.map((c) => `gone   ${c.community}. ${c.name} (${c.notes} notes then)`)].map((line) => (
					<Text key={line} wrap="truncate-end">
						{line}
					</Text>
				))}
			</Group>
			<Group title={`Missing notes written: ${data.written.length}`}>
				{data.written.map((item) => (
					<Text key={item.name} wrap="truncate-end">
						[[{item.name}]] <Text dimColor>is now {item.note}</Text>
					</Text>
				))}
			</Group>
		</Box>
	);
}

export function ConfigView({data}: {data: ConfigShowResult}) {
	const width = Math.max(...data.settings.map((row) => row.key.length));
	return (
		<Box flexDirection="column">
			<Headline title="Engine settings" detail={`precedence: ${data.precedence.join(' > ')}`} />
			<Body>
				<Text dimColor>config file {data.config_file ?? 'none'} ({data.config_found_by})</Text>
			</Body>
			<Box flexDirection="column" paddingLeft={5} marginTop={1}>
				{data.settings.map((row) => (
					<Text key={row.key} wrap="truncate-end">
						{row.key.padEnd(width)}  <Text bold>{row.value ?? '(not set)'}</Text>
						<Text dimColor>  {row.source}</Text>
					</Text>
				))}
			</Box>
		</Box>
	);
}
