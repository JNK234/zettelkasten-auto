/** Views for /related, /suggest and /unlinked (static transcript output). */
import {Box, Text} from 'ink';
import type {LinksSuggestResult, MentionsUnlinkedResult, NoteRelatedResult, NotesSuggestResult} from '../../rpc/schema.js';
import {formatNumber, plural, splitPath, theme} from '../theme.js';
import {Marked} from './Marked.js';
import {Body, Group, Headline} from './Section.js';

function Status({linked}: {linked: boolean}) {
	return linked ? <Text dimColor>linked</Text> : <Text color={theme.warn}>not linked</Text>;
}

/** "0.5 × graph + 0.3 × text + 0.2 × tags" from the weights the engine used (a signal that is off is left out). */
export function formula(weights: Record<string, number>, names: Record<string, string>): string {
	return Object.keys(names)
		.filter((key) => (weights[key] ?? 0) > 0)
		.map((key) => `${Number(weights[key]!.toPrecision(2))} × ${names[key]}`)
		.join(' + ');
}

/** Without the embedding model the text signal is off (and why); vectors the index lacks were computed for this answer. */
function TextSignalNote({signal}: {signal: NoteRelatedResult['text_signal']}) {
	if (!signal.available) {
		return <Text color={theme.warn}>text signal unavailable: {signal.unavailable} (the other signals still rank)</Text>;
	}
	if (signal.uncached > 0) {
		return <Text dimColor>{plural(signal.uncached, 'note')} had no cached vector: embedded for this answer only; `vault embed` caches them</Text>;
	}
	return null;
}

export function RelatedView({data}: {data: NoteRelatedResult}) {
	const {seed} = data;
	const detail = seed.path ?? `no note yet; ${plural(seed.linkers.length, 'note')} link to it`;
	return (
		<Box flexDirection="column">
			<Headline title={`Related to ${seed.label}`} detail={detail} />
			<Body>
				<Text dimColor>
					score = {formula(data.weights, {graph: 'graph', cocitation: 'cocitation', coupling: 'coupling', text: 'text', tags: 'tags'})} · {plural(data.compared, 'note')} compared
				</Text>
				<TextSignalNote signal={data.text_signal} />
			</Body>
			{data.hits.length === 0 ? (
				<Text dimColor>{'     '}nothing related found</Text>
			) : (
				data.hits.map((hit, index) => {
					const {folder} = splitPath(hit.path);
					return (
						<Box key={hit.path} flexDirection="column" marginTop={1} paddingLeft={2}>
							<Text wrap="truncate-end">
								<Text dimColor>{`${index + 1}.`.padStart(3)} </Text>
								<Text bold color={theme.link}>
									{hit.title}
								</Text>
								<Text dimColor>  {folder}</Text>
								<Text>  {hit.score.toFixed(2)}  </Text>
								<Status linked={hit.linked} />
							</Text>
							<Box paddingLeft={4}>
								<Text dimColor>{hit.reason || '(weak: only distant links)'}</Text>
							</Box>
						</Box>
					);
				})
			)}
		</Box>
	);
}

function name(path: string): string {
	return splitPath(path).title;
}

export function LinksView({data}: {data: LinksSuggestResult}) {
	const scope = data.note ? ` with ${name(data.note)}` : data.path ? ` in ${data.path}` : '';
	return (
		<Box flexDirection="column">
			<Headline
				title={`Links to add${scope}`}
				detail={`score = ${formula(data.weights, {shared: 'shared neighbours', cited: 'cited together', tags: 'shared tags', text: 'similar text'})}`}
			/>
			{!data.text_signal.available || data.text_signal.uncached > 0 ? (
				<Body>
					<TextSignalNote signal={data.text_signal} />
				</Body>
			) : null}
			{data.links.length === 0 ? (
				<Text dimColor>{'     '}nothing to suggest</Text>
			) : (
				data.links.map((pair, index) => (
					<Box key={`${pair.a}|${pair.b}`} flexDirection="column" marginTop={1} paddingLeft={2}>
						<Text>
							<Text dimColor>{`${index + 1}.`.padStart(3)} </Text>
							<Text bold color={theme.link}>
								{name(pair.a)}
							</Text>
							<Text>  ↔  </Text>
							<Text bold color={theme.link}>
								{name(pair.b)}
							</Text>
							<Text>  {pair.score.toFixed(2)}</Text>
						</Text>
						<Box paddingLeft={4}>
							<Text dimColor>{pair.reason}</Text>
						</Box>
					</Box>
				))
			)}
		</Box>
	);
}

export function NotesToWriteView({data}: {data: NotesSuggestResult}) {
	return (
		<Box flexDirection="column">
			<Headline title="Notes to write" detail={`${plural(data.total_missing, 'link target')} have no note; most wanted first`} />
			{data.notes.map((note, index) => (
				<Box key={note.name} flexDirection="column" marginTop={1} paddingLeft={2}>
					<Text wrap="truncate-end">
						<Text dimColor>{`${index + 1}.`.padStart(3)} </Text>
						<Text bold color={theme.error}>
							{note.name}
						</Text>
						<Text dimColor>
							{'  '}
							{plural(note.linking_notes, 'note')} link here
							{note.mentions > 0 ? ` · ${formatNumber(note.mentions)} more mention it` : ''}
						</Text>
					</Text>
					{note.about.length > 0 ? (
						<Text dimColor wrap="truncate-end">
							{'    '}about: {note.about.join(', ')}
						</Text>
					) : null}
				</Box>
			))}
			{data.same_idea.length > 0 ? (
				<Group title="Overlapping names (the same idea twice, or a topic and a narrower one)">
					{data.same_idea.slice(0, 5).map((group) => (
						<Text key={group[0]!.name}>
							{group.map((item) => `${item.name} (${item.linking_notes})`).join(' · ')}
						</Text>
					))}
				</Group>
			) : null}
			{data.existing.length > 0 ? (
				<Group title="Probably an existing note">
					{data.existing.slice(0, 5).map((item) => (
						<Text key={item.name}>
							[[{item.name}]] <Text dimColor>({plural(item.linking_notes, 'note')})</Text> → <Text color={theme.link}>{item.path}</Text>
						</Text>
					))}
				</Group>
			) : null}
			<Text dimColor>{'\n     '}Draft one from the shell: vault suggest &lt;vault&gt; --draft "&lt;name&gt;"</Text>
		</Box>
	);
}

export function UnlinkedView({data}: {data: MentionsUnlinkedResult}) {
	const scope = data.note ? name(data.note) : 'any note';
	const verb = data.notes === 1 ? 'does' : 'do';
	return (
		<Box flexDirection="column">
			<Headline title={`Unlinked mentions of ${scope}`} detail={`${plural(data.total, 'mention')} in ${plural(data.notes, 'note')} that ${verb} not link to it yet`} />
			{data.targets.map((target) => (
				<Box key={target.path} flexDirection="column" marginTop={1} paddingLeft={2}>
					<Text>
						<Text bold color={theme.link}>
							{target.title}
						</Text>
						<Text dimColor>
							{'  '}
							{plural(target.notes, 'note')}
							{target.names.some((n) => n !== target.title) ? `; as ${target.names.map((n) => JSON.stringify(n)).join(', ')}` : ''}
						</Text>
					</Text>
					{target.mentions.map((m) => {
						const before = flat(m.before);
						const shown = flat(m.text);
						return (
							<Box key={`${m.source}:${m.start}`} flexDirection="column" paddingLeft={2}>
								<Text dimColor>
									{m.source}:{m.line}
								</Text>
								<Box paddingLeft={2}>
									<Marked
										dim
										text={before + shown + flat(m.after)}
										marks={[{start: [...before].length, end: [...before].length + [...shown].length, kind: 'exact'}]}
									/>
								</Box>
							</Box>
						);
					})}
					{target.total > target.mentions.length ? (
						<Text dimColor>
							{'  '}… {formatNumber(target.total - target.mentions.length)} more
						</Text>
					) : null}
				</Box>
			))}
			{data.targets.length === 0 ? <Text dimColor>{'     '}none: every mention is already a link</Text> : null}
		</Box>
	);
}

/** Whitespace runs (line breaks inside a mention) as single spaces, keeping a leading/trailing space. */
function flat(text: string): string {
	return text.replaceAll(/\s+/g, ' ');
}
