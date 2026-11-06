import {Box, Text} from 'ink';
import type {VaultStatsResult} from '../../rpc/schema.js';
import {formatNumber, plural} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';

export function StatsView({data}: {data: VaultStatsResult}) {
	const {links} = data;
	return (
		<Box flexDirection="column">
			<Headline title="Vault stats" />
			<Body>
				<Text>
					{plural(data.notes, 'note')} · {plural(data.attachments, 'attachment')} · {formatNumber(data.notes_with_frontmatter)} with
					frontmatter
				</Text>
				<Text>
					{plural(links.total, 'link')}: {formatNumber(links.resolved)} resolved · {formatNumber(links.unresolved)} unresolved (
					{plural(links.unresolved_targets, 'target')}) · {formatNumber(links.ambiguous)} ambiguous · {plural(links.embeds, 'embed')}
				</Text>
				<Text>
					{plural(data.note_edges, 'note→note edge')} · {plural(data.tags.distinct, 'tag')} · {plural(data.headings, 'heading')} ·{' '}
					{plural(data.blocks, 'block')} · {plural(data.aliases, 'alias', 'aliases')}
				</Text>
			</Body>
			{data.tags.top.length > 0 ? (
				<Group title="Top tags">
					<Text>{data.tags.top.map((tag) => `#${tag.tag} ${formatNumber(tag.notes)}`).join('  ·  ')}</Text>
				</Group>
			) : null}
			{data.most_linked.length > 0 ? (
				<Group title="Most linked (linking notes)">
					{data.most_linked.map((note) => (
						<Text key={note.path}>
							<Text>{formatNumber(note.linking_notes).padStart(4)}  </Text>
							<NotePath path={note.path} />
						</Text>
					))}
				</Group>
			) : null}
			{data.most_wanted.length > 0 ? (
				<Group title="Most wanted missing notes (linking notes)">
					{data.most_wanted.map((wanted) => (
						<Text key={wanted.target}>
							<Text>{formatNumber(wanted.linking_notes).padStart(4)}  </Text>
							<Text color="yellow">{wanted.target}</Text>
						</Text>
					))}
				</Group>
			) : null}
		</Box>
	);
}
