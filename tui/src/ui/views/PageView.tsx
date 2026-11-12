import {Box, Text} from 'ink';
import type {ReactNode} from 'react';
import type {PageCheck, PageCheckResult} from '../../rpc/schema.js';
import {plural, splitPath, theme} from '../theme.js';
import {Body, Headline, NotePath} from './Section.js';
import {ViolationLine} from './Violations.js';

function Problem({children}: {children: ReactNode}) {
	return (
		<Text>
			<Text color={theme.error}>✗</Text> {children}
		</Text>
	);
}

/** One page: ready, or what needs attention and the command that fixes it. */
export function PageCheckLines({page, maxRows = 8}: {page: PageCheck; maxRows?: number}) {
	const name = splitPath(page.path).title;
	const outOfDate = page.blocks.filter((block) => block.status === 'out-of-date');
	const missing = page.blocks.filter((block) => block.status === 'missing');
	const needsRefresh = outOfDate.length > 0 || missing.length > 0 || page.changed === null || (page.changed?.length ?? 0) > 0 || page.index !== 'current';
	return (
		<Box flexDirection="column" marginTop={1}>
			<Text>
				<NotePath path={page.path} />
				<Text dimColor>
					{'  '}
					{page.title}
					{page.managed ? ` · ${page.refreshed_at ? `refreshed ${page.refreshed_at}` : 'no refresh recorded'}` : ''}
				</Text>
			</Text>
			<Box flexDirection="column" paddingLeft={2}>
				{page.renamed_from ? <Text dimColor>= renamed from {page.renamed_from}: its record followed it</Text> : null}
				{page.excluded.length > 0 ? <Text dimColor>= {plural(page.excluded.length, 'zettel')} excluded from it</Text> : null}
				{page.problems.map((problem) => (
					<Problem key={problem}>{problem}</Problem>
				))}
				{page.managed && page.ready ? (
					<Text>
						<Text color={theme.ok}>✓</Text> ready to write: blocks up to date, zettels as recorded, nothing behind, in the index, no
						violations
					</Text>
				) : null}
				{missing.map((block) => (
					<Problem key={block.name}>{block.name} block missing (its markers are gone)</Problem>
				))}
				{outOfDate.map((block) => (
					<Box key={block.name} flexDirection="column">
						<Problem>
							{block.name} block out of date <Text dimColor>(+{block.added.length} −{block.removed.length} lines)</Text>
						</Problem>
						{block.removed.slice(0, maxRows).map((line) => (
							<Text key={`-${line}`} color={theme.error}>
								{'    - '}
								{line}
							</Text>
						))}
						{block.added.slice(0, maxRows).map((line) => (
							<Text key={`+${line}`} color={theme.ok}>
								{'    + '}
								{line}
							</Text>
						))}
					</Box>
				))}
				{page.managed && page.changed === null ? <Problem>no record of its zettels: refresh it once so changes can be tracked</Problem> : null}
				{page.changed && page.changed.length > 0 ? (
					<>
						<Problem>{plural(page.changed.length, 'zettel')} changed since the last refresh</Problem>
						{page.changed.slice(0, maxRows).map((change) => (
							<Box key={change.path} paddingLeft={4}>
								<Box width={14} flexShrink={0}>
									<Text color={theme.warn}>{change.status}</Text>
								</Box>
								<NotePath path={change.path} />
							</Box>
						))}
					</>
				) : null}
				{page.behind.length > 0 ? (
					<>
						<Problem>behind: {plural(page.behind.length, 'zettel')} with two or more of its tags or links in common, not on it</Problem>
						{page.behind.slice(0, maxRows).map((row) => (
							<Box key={row.path} flexDirection="column" paddingLeft={4}>
								<NotePath path={row.path} />
								<Box paddingLeft={2}>
									<Text dimColor>
										{[row.tags.length > 0 ? `tags ${row.tags.join(', ')}` : '', row.links.length > 0 ? `links ${row.links.join(', ')}` : '']
											.filter(Boolean)
											.join(' · ')}
									</Text>
								</Box>
							</Box>
						))}
						{page.behind.length > maxRows ? <Text dimColor>    … {plural(page.behind.length - maxRows, 'more zettel')}</Text> : null}
					</>
				) : null}
				{page.managed && page.index !== 'current' ? <Problem>index line {page.index === 'missing' ? 'missing' : 'out of date'}</Problem> : null}
				{page.violations.map((violation) => (
					<ViolationLine key={`${violation.line} ${violation.code}`} violation={violation} />
				))}
				{page.managed && needsRefresh ? (
					<Text>
						→ <Text color={theme.accent}>/page refresh @{name} write</Text>
					</Text>
				) : null}
				{page.behind.length > 0 ? (
					<Box>
						<Text>→ </Text>
						<Text>
							link the ones that belong in its text, then refresh; <Text color={theme.accent}>/page exclude @{name} @zettel… write</Text> for
							the others
						</Text>
					</Box>
				) : null}
			</Box>
		</Box>
	);
}

export function PageCheckView({data, one}: {data: PageCheckResult; one: boolean}) {
	const shown = data.pages.filter((page) => page.managed || one);
	const unmanaged = data.pages.filter((page) => !page.managed).length;
	return (
		<Box flexDirection="column">
			<Headline title="Page check" detail={`${data.format.name} · ${plural(data.managed, 'managed page')}`} />
			<Body>
				<Text>
					<Text color={data.ready ? theme.ok : theme.error}>{data.ready ? '✓' : '✗'}</Text> {data.ready_pages} of{' '}
					{plural(data.managed, 'managed page')} ready to write
				</Text>
				{!one && unmanaged > 0 ? <Text dimColor>{plural(unmanaged, 'page')} without vault blocks (not made by /page new)</Text> : null}
			</Body>
			<Box flexDirection="column" paddingLeft={5}>
				{shown.map((page) => (
					<PageCheckLines key={page.path} page={page} />
				))}
			</Box>
		</Box>
	);
}
