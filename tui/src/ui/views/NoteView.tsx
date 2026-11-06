import {Box, Text} from 'ink';
import type {NoteGetResult, OutgoingLink} from '../../rpc/schema.js';
import {formatNumber, plural, splitPath, theme} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';

function subpathText(link: {heading_path: string[]; block_id: string | null}): string {
	if (link.block_id) {
		return `#^${link.block_id}`;
	}

	return link.heading_path.length > 0 ? `#${link.heading_path.join('#')}` : '';
}

function propertyValue(value: unknown): string {
	if (Array.isArray(value)) {
		return value.map(String).join(', ');
	}

	return typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value);
}

function LinkLine({link}: {link: OutgoingLink}) {
	const subpath = subpathText(link);
	const embed = link.is_embed ? <Text dimColor>embed </Text> : null;
	const line = <Text dimColor>  L{link.line}</Text>;
	if (link.status === 'ambiguous') {
		return (
			<Text>
				<Text color={theme.warn}>? </Text>
				{embed}
				{link.target}
				{subpath}
				<Text dimColor>  ambiguous: {link.candidates.join(' | ')}</Text>
				{line}
			</Text>
		);
	}

	if (link.status === 'unresolved') {
		return (
			<Text>
				<Text color={theme.error}>✗ </Text>
				{embed}
				<Text color={theme.error}>{link.target}</Text>
				{subpath}
				<Text dimColor>  not in vault</Text>
				{line}
			</Text>
		);
	}

	const target = link.matched_by === 'self' ? '(this note)' : splitPath(link.target_path ?? link.target).title;
	return (
		<Text>
			<Text color={theme.ok}>→ </Text>
			{embed}
			<Text color={theme.link}>{target}</Text>
			{subpath ? <Text color={link.subpath_status === 'missing' ? theme.error : undefined}>{subpath}</Text> : null}
			{link.subpath_status === 'missing' ? <Text color={theme.error}>  (not found)</Text> : null}
			{link.matched_by === 'alias' ? <Text dimColor>  via alias "{link.target}"</Text> : null}
			{line}
		</Text>
	);
}

export function NoteView({note, matchedFrom}: {note: NoteGetResult; matchedFrom?: string}) {
	const resolved = note.links_out.filter((link) => link.status === 'resolved').length;
	const properties = Object.entries(note.properties).filter(([key]) => key !== 'tags' && key !== 'aliases');
	const tags = [...new Set(note.tags.map((tag) => `#${tag.name}`))];
	return (
		<Box flexDirection="column">
			<Headline title={note.title} detail={note.path} />
			<Body>
				{matchedFrom ? <Text dimColor>best match for "{matchedFrom}"</Text> : null}
				<Text>
					{tags.length > 0 ? tags.join(' ') : <Text dimColor>no tags</Text>}
					<Text dimColor>
						{'  ·  '}
						{plural(note.line_count, 'line')} · {plural(note.headings.length, 'heading')}
					</Text>
				</Text>
				{note.aliases.length > 0 ? <Text>aliases: {note.aliases.join(', ')}</Text> : null}
				{properties.map(([key, value]) => (
					<Text key={key}>
						<Text dimColor>{key}: </Text>
						{propertyValue(value)}
					</Text>
				))}
				{note.frontmatter_error ? <Text color={theme.error}>frontmatter error: {note.frontmatter_error}</Text> : null}
			</Body>
			{note.headings.length > 0 ? (
				<Group title="Outline">
					{note.headings.map((heading) => (
						<Text key={`${heading.line}`}>
							{'  '.repeat(heading.level - 1)}
							<Text dimColor>{'#'.repeat(heading.level)} </Text>
							{heading.text}
						</Text>
					))}
				</Group>
			) : null}
			<Group
				title={`Links out (${formatNumber(note.links_out.length)}${note.links_out.length > 0 ? `: ${formatNumber(resolved)} resolved` : ''})`}
			>
				{note.links_out.length > 0 ? (
					note.links_out.map((link, index) => <LinkLine key={`${index}:${link.raw}`} link={link} />)
				) : (
					<Text dimColor>none</Text>
				)}
			</Group>
			<Group title={`Links in (${formatNumber(note.links_in.length)})`}>
				{note.links_in.length > 0 ? (
					note.links_in.map((link, index) => (
						<Text key={`${index}:${link.source}:${link.line}`}>
							<Text color={theme.ok}>← </Text>
							<NotePath path={link.source} />
							{subpathText(link) ? (
								<Text color={link.subpath_status === 'missing' ? theme.error : undefined}>  {subpathText(link)}</Text>
							) : null}
							{link.subpath_status === 'missing' ? <Text color={theme.error}>  (not found here)</Text> : null}
							<Text dimColor>  L{link.line}</Text>
						</Text>
					))
				) : (
					<Text dimColor>none: this note is an orphan</Text>
				)}
			</Group>
		</Box>
	);
}
