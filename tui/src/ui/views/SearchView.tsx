/**
 * /search results: an interactive list with a preview.
 *
 *   list     ↑/↓ (PgUp/PgDn, Home/End) move · Enter preview · o open in Obsidian · Esc/q close
 *   preview  ↑/↓ (PgUp/PgDn) scroll · n/N next/previous match · o open in Obsidian · Esc back to the list
 *
 * When closed, a static summary (the top results) goes to the transcript.
 */
import {Box, Text, useInput, useStdout} from 'ink';
import {type ReactNode, useEffect, useRef, useState} from 'react';
import type {VaultApi} from '../../rpc/api.js';
import type {NotePreviewResult, SearchHit, SearchQueryResult} from '../../rpc/schema.js';
import {formatNumber, plural, splitPath, theme} from '../theme.js';
import {Marked} from './Marked.js';
import {Body, Headline} from './Section.js';

const FIELD_NAMES: Record<string, string> = {title: 'title', aliases: 'alias', headings: 'heading', tags: 'tag', body: 'body'};
const SUMMARY_HITS = 10;

function orderText(order: SearchQueryResult['order']): string {
	if (order === 'path') {
		return 'by path (no words to rank by)';
	}

	return order === 'score' ? 'by score' : 'exact matches first, then score';
}

function whereText(hit: SearchHit): string {
	return hit.matches
		.map((match) => {
			const fields = match.fields.map((f) => `${FIELD_NAMES[f.field] ?? f.field}${f.count > 1 ? ` ×${f.count}` : ''}`).join(', ');
			const label = match.exact ? match.clause : `${match.clause} ≈ ${match.term ?? '?'} (${plural(match.edits, 'edit')})`;
			return `${label}: ${fields}`;
		})
		.join(' · ');
}

function SearchHeader({result}: {result: SearchQueryResult}) {
	return (
		<Box flexDirection="column">
			<Headline title={`Search “${result.query}”`} detail={`${formatNumber(result.total)} of ${plural(result.notes, 'note')} · ${orderText(result.order)}`} />
			{result.warnings.map((warning) => (
				<Text key={warning} color={theme.warn}>
					{'     '}note: {warning}
				</Text>
			))}
		</Box>
	);
}

function HitLines({hit, rank, selected}: {hit: SearchHit; rank: number; selected: boolean}) {
	const {folder} = splitPath(hit.path);
	const snippet = hit.snippet;
	return (
		<Box flexDirection="column" marginTop={1}>
			<Text wrap="truncate-end">
				<Text color={theme.accent}>{selected ? '❯ ' : '  '}</Text>
				<Text dimColor>{`${rank}.`.padStart(3)} </Text>
				<Text bold inverse={selected}>
					<Marked text={hit.title} marks={hit.title_marks} />
				</Text>
				<Text dimColor>  {hit.matches.length > 0 ? hit.score.toFixed(2) : 'filters only'}</Text>
			</Text>
			<Text wrap="truncate-end" dimColor>
				{'      '}
				{folder || '(vault root)'}
				{hit.matches.length > 0 ? `  ·  ${whereText(hit)}` : ''}
			</Text>
			{snippet ? (
				<Text wrap="truncate-end">
					{'      '}
					{snippet.truncated_start ? <Text dimColor>… </Text> : null}
					<Marked text={snippet.text} marks={snippet.marks} />
					{snippet.truncated_end ? <Text dimColor> …</Text> : null}
				</Text>
			) : null}
		</Box>
	);
}

/** What stays in the transcript after the browser is closed. */
export function SearchSummary({result}: {result: SearchQueryResult}) {
	const shown = result.hits.slice(0, SUMMARY_HITS);
	const more = result.total - shown.length;
	const tags = result.facets.tags.slice(0, 5);
	const folders = result.facets.folders.slice(0, 5);
	return (
		<Box flexDirection="column">
			<SearchHeader result={result} />
			{shown.length === 0 ? (
				<Body>
					<Text dimColor>no notes match</Text>
				</Body>
			) : (
				shown.map((hit, index) => <HitLines key={hit.path} hit={hit} rank={index + 1} selected={false} />)
			)}
			{more > 0 ? (
				<Text dimColor>
					{'     '}… {formatNumber(more)} more (narrow the query, e.g. add tag:… or path:…)
				</Text>
			) : null}
			{tags.length > 0 ? (
				<Text dimColor wrap="truncate-end">
					{'     '}tags: {tags.map((t) => `#${t.name} ${t.notes}`).join(' · ')}
				</Text>
			) : null}
			{folders.length > 0 ? (
				<Text dimColor wrap="truncate-end">
					{'     '}folders: {folders.map((f) => `${f.name || '(vault root)'} ${f.notes}`).join(' · ')}
				</Text>
			) : null}
		</Box>
	);
}

type PreviewState = {path: string; data?: NotePreviewResult; error?: string; top: number};
type Notice = {text: string; ok: boolean; url?: string};

/** How many screen rows a line takes when wrapped at `width` columns. */
function rowsOf(text: string, width: number): number {
	return Math.max(1, Math.ceil([...text].length / Math.max(1, width)));
}

/** The lines from `top` that fit in `height` rows (always at least one). */
export function fitLines<T extends {text: string}>(lines: T[], top: number, height: number, width: number): T[] {
	const shown: T[] = [];
	let used = 0;
	for (const line of lines.slice(top)) {
		const rows = rowsOf(line.text, width);
		if (shown.length > 0 && used + rows > height) {
			break;
		}

		shown.push(line);
		used += rows;
	}

	return shown;
}

function PreviewPane({preview, height, width}: {preview: PreviewState; height: number; width: number}) {
	if (preview.error) {
		return <Text color={theme.error}>{'     '}{preview.error}</Text>;
	}

	if (!preview.data) {
		return <Text dimColor>{'     '}Loading {preview.path}…</Text>;
	}

	const {data} = preview;
	const numberWidth = String(data.lines.length).length;
	const visible = fitLines(data.lines, preview.top, height, width - numberWidth - 3); // "123 │ " takes the rest
	const position = `lines ${preview.top + 1}–${preview.top + visible.length} of ${data.lines.length}`;
	return (
		<Box flexDirection="column">
			<Headline title={data.title} detail={data.path} />
			<Text dimColor>
				{'     '}
				{data.marks > 0 ? plural(data.marks, 'match', 'matches') : 'no matches in the text (title, tags or properties only)'} · {position}
			</Text>
			<Box flexDirection="column" marginTop={1}>
				{visible.map((line) => (
					// A row: the gutter keeps its width and wrapped text stays in the text column.
					<Box key={line.number}>
						<Text dimColor>{`${String(line.number).padStart(numberWidth)} │ `}</Text>
						<Box flexShrink={1}>
							<Marked text={line.text} marks={line.marks} />
						</Box>
					</Box>
				))}
			</Box>
		</Box>
	);
}

export type SearchBrowserProps = {
	result: SearchQueryResult;
	api: VaultApi;
	openNote: (path: string) => Promise<string>;
	done: (summary: ReactNode) => void;
};

export function SearchBrowser({result, api, openNote, done}: SearchBrowserProps) {
	const {stdout} = useStdout();
	const terminal = stdout as NodeJS.WriteStream;
	const rows = terminal.rows && terminal.rows > 0 ? terminal.rows : 24;
	const columns = terminal.columns && terminal.columns > 0 ? terminal.columns : 100;
	const pageSize = Math.max(2, Math.floor((rows - 9) / 4)); // each hit takes about 4 rows
	const previewHeight = Math.max(6, rows - 10);
	const [mode, setMode] = useState<'list' | 'preview'>('list');
	const [selected, setSelected] = useState(0);
	const [preview, setPreview] = useState<PreviewState | undefined>();
	const [notice, setNotice] = useState<Notice | undefined>();
	const hits = result.hits;
	const count = hits.length;
	const current = hits[selected];

	// Load the preview of the selected note when it is shown. The requested path lives in a ref,
	// not in the effect's dependencies: otherwise setting the "loading" state would re-run the
	// effect and drop the response in flight. A response is used only if its note is still shown.
	const requested = useRef<string | undefined>(undefined);
	useEffect(() => {
		if (mode !== 'preview' || !current || requested.current === current.path) {
			return;
		}

		const {path} = current;
		requested.current = path;
		setPreview({path, top: 0});
		api.preview({path, query: result.query}).then(
			(data) => {
				const first = data.first_match_line ?? 1;
				setPreview((state) => (state?.path === path ? {path, data, top: Math.max(0, first - 3)} : state));
			},
			(error: unknown) => {
				requested.current = undefined; // allow a retry with Enter
				setPreview((state) => (state?.path === path ? {path, top: 0, error: error instanceof Error ? error.message : String(error)} : state));
			},
		);
	}, [api, current, mode, result.query]);

	const open = (path: string) => {
		openNote(path).then(
			(url) => {
				setNotice({text: `Opened ${path} in Obsidian`, ok: true, url});
			},
			(error: unknown) => {
				setNotice({text: `Could not open Obsidian: ${error instanceof Error ? error.message : String(error)}`, ok: false});
			},
		);
	};

	const scrollPreview = (delta: number) => {
		setPreview((state) => {
			if (!state?.data) {
				return state;
			}

			const last = Math.max(0, state.data.lines.length - previewHeight);
			return {...state, top: Math.min(last, Math.max(0, state.top + delta))};
		});
	};

	const jumpToMatch = (forward: boolean) => {
		setPreview((state) => {
			if (!state?.data) {
				return state;
			}

			const here = state.top + 3; // the match is shown 3 lines below the top
			const marked = state.data.lines.filter((line) => line.marks.length > 0).map((line) => line.number - 1);
			const target = forward ? marked.find((index) => index > here) : marked.findLast((index) => index < here);
			if (target === undefined) {
				return state;
			}

			const last = Math.max(0, state.data.lines.length - previewHeight);
			return {...state, top: Math.min(last, Math.max(0, target - 3))};
		});
	};

	// Declared after nothing else takes input: the key hints are shown once this
	// component is subscribed (see `ready`), so what the hints promise works.
	useInput((input, key) => {
		if (mode === 'preview') {
			if (key.escape) {
				setMode('list');
				setNotice(undefined);
			} else if (key.upArrow || key.downArrow) {
				scrollPreview(key.upArrow ? -1 : 1);
			} else if (key.pageUp || key.pageDown) {
				scrollPreview((key.pageUp ? -1 : 1) * previewHeight);
			} else if (input === 'n') {
				jumpToMatch(true);
			} else if (input === 'N' || input === 'p') {
				jumpToMatch(false);
			} else if (input === 'o' && current) {
				open(current.path);
			}

			return;
		}

		if (key.escape || input === 'q') {
			done(<SearchSummary result={result} />);
		} else if ((key.upArrow || key.downArrow) && count > 0) {
			setSelected((index) => (index + (key.upArrow ? count - 1 : 1)) % count);
			setNotice(undefined);
		} else if ((key.pageUp || key.pageDown) && count > 0) {
			setSelected((index) => Math.min(count - 1, Math.max(0, index + (key.pageUp ? -pageSize : pageSize))));
		} else if (key.home) {
			setSelected(0);
		} else if (key.end) {
			setSelected(Math.max(0, count - 1));
		} else if (key.return && current) {
			setMode('preview');
			setNotice(undefined);
		} else if (input === 'o' && current) {
			open(current.path);
		}
	});

	// Runs after useInput's own effect (effects run in declaration order): from now on keys arrive here.
	const [ready, setReady] = useState(false);
	useEffect(() => {
		setReady(true);
	}, []);

	const firstShown = Math.min(Math.max(0, selected - pageSize + 1), Math.max(0, count - pageSize), selected);
	const page = hits.slice(firstShown, firstShown + pageSize);
	const keys =
		mode === 'preview'
			? '↑↓ scroll · PgUp/PgDn page · n/N next/previous match · o open in Obsidian · Esc back to the list'
			: '↑↓ move · Enter preview · o open in Obsidian · Esc close';
	return (
		<Box flexDirection="column">
			{mode === 'list' ? (
				<>
					<SearchHeader result={result} />
					{page.map((hit, index) => (
						<HitLines key={hit.path} hit={hit} rank={firstShown + index + 1} selected={firstShown + index === selected} />
					))}
					<Text dimColor>
						{'\n     '}
						{count > 0 ? `${selected + 1} of ${formatNumber(count)}` : 'no results'}
						{result.total > count ? ` shown (${formatNumber(result.total)} match)` : ''}
					</Text>
				</>
			) : preview ? (
				<PreviewPane preview={preview} height={previewHeight} width={columns} />
			) : null}
			{notice ? (
				<Box flexDirection="column" paddingLeft={5}>
					<Text color={notice.ok ? theme.ok : theme.error}>
						{notice.ok ? '↗ ' : '✗ '}
						{notice.text}
					</Text>
					{notice.url ? <Text dimColor>{notice.url}</Text> : null}
				</Box>
			) : null}
			<Box borderStyle="round" borderColor="gray" paddingX={1}>
				<Text dimColor>{ready ? keys : ' '}</Text>
			</Box>
		</Box>
	);
}

/** A query the engine could not parse: the message, the query with a caret under the problem, and the hint. */
export function QueryErrorView({query, message, start, end, hint}: {query: string; message: string; start?: number | null; end?: number | null; hint?: string | null}) {
	const caret = start === undefined || start === null ? undefined : ' '.repeat(start) + '^'.repeat(Math.max(1, (end ?? start + 1) - start));
	return (
		<Box flexDirection="column">
			<Text>
				<Text color={theme.error}>⏺ </Text>
				<Text color={theme.error}>Invalid query: {message}</Text>
			</Text>
			{caret === undefined ? null : (
				<>
					<Text>{'     '}{query}</Text>
					<Text color={theme.error}>{'     '}{caret}</Text>
				</>
			)}
			{hint ? <Text dimColor>{'     '}{hint}</Text> : null}
		</Box>
	);
}
