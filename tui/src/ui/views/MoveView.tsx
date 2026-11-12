import {Box, Text} from 'ink';
import type {RelocationResult} from '../../rpc/schema.js';
import {plural, theme} from '../theme.js';
import {Group, NotePath} from './Section.js';
import {WriteView} from './WriteView.js';

const WRITE_HINT = 'add write to the command to write it';

/** note.move / note.merge: the write, where notes went, the links re-spelled, and the self-links dropped. */
export function MoveView({result, maxSelfLinks = 8}: {result: RelocationResult; maxSelfLinks?: number}) {
	const notes = new Set(result.rewritten.map((line) => line.split(':')[0]));
	const extra = (
		<>
			{Object.entries(result.moved).map(([from, to]) => (
				<Text key={from}>
					{result.operation === 'merge' ? 'merged ' : 'moved '}
					<NotePath path={from} />
					<Text dimColor> → </Text>
					<NotePath path={to} />
				</Text>
			))}
			<Text>
				links: {plural(result.rewritten.length, 'link')} re-spelled in {plural(notes.size, 'note')}
				<Text dimColor>, every link that reached it still does</Text>
			</Text>
			{result.now_resolving.length > 0 ? (
				<Text color={theme.warn}>{plural(result.now_resolving.length, 'link')} that matched no note reach it now</Text>
			) : null}
		</>
	);
	const after =
		result.self_links.length > 0 ? (
			<Group title={`Self-links dropped (${result.self_links.length}): they would point at the note they are in`}>
				{result.self_links.slice(0, maxSelfLinks).map((link) => (
					<Box key={`${link.path}:${link.line}:${link.link}`}>
						<Text dimColor>L{link.line} </Text>
						<Text>{link.link}</Text>
						<Text dimColor> → </Text>
						<Text color={theme.warn}>{link.action === 'line removed' || link.action === 'removed' ? link.action : link.result}</Text>
					</Box>
				))}
				{result.self_links.length > maxSelfLinks ? <Text dimColor>… {plural(result.self_links.length - maxSelfLinks, 'more')}</Text> : null}
			</Group>
		) : null;
	return <WriteView result={result} writeHint={WRITE_HINT} extra={extra} after={after} maxFiles={5} maxLines={24} />;
}
