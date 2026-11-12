/**
 * /explore: the engine's explore lenses as an interactive list, one section per lens.
 *
 *   list    ↑/↓ move · Enter open the result · k keep · d dismiss · c clear the mark · l link it (dry run first)
 *           p pivot: explore from this result · b back to the previous start · r run again · f run again with fresh · Esc
 *   detail  the full evidence, the notes, what link it would join; the same keys; Esc back to the list
 *   link    the dry run's diff: w writes it (both notes, through the safe write path) · Esc cancels
 *
 * Marks and links go to the engine (explore.mark, explore.link): its memory, never a note except through link it.
 * When closed, a short summary of the last run goes to the transcript.
 */
import {Box, Text, useInput, useStdout} from 'ink';
import {type ReactNode, useEffect, useState} from 'react';
import type {VaultApi} from '../../rpc/api.js';
import type {ExploreItem, ExploreLinkResult, ExploreRunResult, ExploreSection} from '../../rpc/schema.js';
import {plural, splitPath, theme} from '../theme.js';
import {Headline} from './Section.js';
import {WriteView} from './WriteView.js';

type Mark = 'keep' | 'dismiss' | null;

export type ExploreStart = {note?: string; label: string};

function startLabel(data: ExploreRunResult): string {
	if (data.start === null) {
		return 'where to start';
	}

	return data.start_kind === 'missing' ? data.start : splitPath(data.start).title;
}

function space(data: ExploreRunResult): string {
	const counts = {deepen: 0, discover: 0};
	for (const section of data.sections) {
		counts[section.kind === 'deepen' ? 'deepen' : 'discover'] += section.results.length;
	}

	return data.start === null
		? `proposals from ${plural(data.sections.length, 'lens', 'lenses')}`
		: `${plural(data.sections.length, 'lens', 'lenses')} · deepen ${counts.deepen}, discover ${counts.discover}`;
}

function rate(section: ExploreSection): string {
	const marked = section.kept + section.dismissed;
	return marked === 0 ? 'no marks yet' : `kept ${section.kept} of ${marked} (${Math.round((100 * section.kept) / marked)}%)`;
}

function markLabel(mark: Mark): string {
	return mark === 'keep' ? '  [kept]' : mark === 'dismiss' ? '  [dismissed]' : '';
}

/** Where pivoting on a result goes: its proposal's start, else its (first) note. */
export function pivotTarget(item: ExploreItem): string | undefined {
	if (item.start_at) {
		return item.start_at.startsWith('[[') ? item.start_at.slice(2, -2) : item.start_at;
	}

	if (item.kind === 'missing') {
		return item.target.slice(2, -2);
	}

	return item.notes[0];
}

function ExploreHeader({data}: {data: ExploreRunResult}) {
	const reused = data.numbers.source === 'reused';
	return (
		<Box flexDirection="column">
			<Headline title={`Explore ${startLabel(data)}`} detail={`${space(data)}${data.fresh ? ' · fresh' : ''}`} />
			{reused ? (
				<Text dimColor wrap="truncate-end">
					{'     '}vault-wide numbers from {data.numbers.computed_at}: {plural(data.numbers.changed_since ?? 0, 'note')} changed since
				</Text>
			) : null}
			{data.text_signal.available ? null : (
				<Text color={theme.warn} wrap="truncate-end">
					{'     '}text signal unavailable: {data.text_signal.unavailable}
				</Text>
			)}
		</Box>
	);
}

type Row =
	| {kind: 'section'; section: ExploreSection; number: number}
	| {kind: 'empty'; section: ExploreSection}
	| {kind: 'item'; item: ExploreItem; index: number};

function rowsOf(data: ExploreRunResult): Row[] {
	const rows: Row[] = [];
	let index = 0;
	data.sections.forEach((section, number) => {
		rows.push({kind: 'section', section, number: number + 1});
		if (section.results.length === 0) {
			rows.push({kind: 'empty', section});
		}

		for (const item of section.results) {
			rows.push({kind: 'item', item, index});
			index += 1;
		}
	});
	return rows;
}

const HEIGHT = {section: 2, empty: 1, item: 2} as const;

/** The rows to show: the selected item visible, as much context around it as fits in `height`. */
export function windowOf(rows: Row[], selected: number, height: number): Row[] {
	const at = Math.max(0, rows.findIndex((row) => row.kind === 'item' && row.index === selected));
	let top = at;
	let used = HEIGHT[rows[at]?.kind ?? 'item'];
	while (top > 0 && used + HEIGHT[rows[top - 1]!.kind] <= Math.ceil(height / 2)) {
		top -= 1;
		used += HEIGHT[rows[top]!.kind];
	}

	let bottom = at + 1;
	while (bottom < rows.length && used + HEIGHT[rows[bottom]!.kind] <= height) {
		used += HEIGHT[rows[bottom]!.kind];
		bottom += 1;
	}

	while (top > 0 && used + HEIGHT[rows[top - 1]!.kind] <= height) {
		top -= 1;
		used += HEIGHT[rows[top]!.kind];
	}

	return rows.slice(top, bottom);
}

function SectionRow({section, number}: {section: ExploreSection; number: number}) {
	return (
		<Box marginTop={1}>
			<Text wrap="truncate-end">
				<Text bold>{`${number}. ${section.title}`}</Text>
				<Text dimColor>
					{'  '}
					{section.kind} · {section.results.length} of {section.found}
					{section.hidden > 0 ? `, ${section.hidden} seen before` : ''}
					{section.duplicates > 0 ? `, ${plural(section.duplicates, 'near-duplicate')}` : ''} · {rate(section)}
				</Text>
			</Text>
		</Box>
	);
}

function ItemRow({item, selected, mark}: {item: ExploreItem; selected: boolean; mark: Mark}) {
	return (
		<Box flexDirection="column">
			<Text wrap="truncate-end">
				<Text color={theme.accent}>{selected ? '❯ ' : '  '}</Text>
				<Text dimColor>{item.ref.padStart(5)}  </Text>
				{mark ? <Text color={mark === 'keep' ? theme.ok : theme.warn}>{mark === 'keep' ? '[kept] ' : '[dismissed] '}</Text> : null}
				<Text bold inverse={selected}>
					{item.title}
				</Text>
			</Text>
			<Text dimColor wrap="truncate-end">
				{'         '}
				{item.lens} · {item.evidence}
			</Text>
		</Box>
	);
}

function Detail({item, section, mark}: {item: ExploreItem; section: ExploreSection; mark: Mark}) {
	return (
		<Box flexDirection="column">
			<Headline title={`${item.ref}  ${item.title}`} detail={`${section.title} · ${section.kind}${markLabel(mark)}`} />
			<Box flexDirection="column" paddingLeft={5} marginTop={1}>
				<Text dimColor>{section.what}</Text>
				<Box marginTop={1}>
					<Text>
						<Text color={theme.accent}>{item.lens}</Text> · {item.evidence}
					</Text>
				</Box>
				{item.notes.length > 0 ? (
					<Box flexDirection="column" marginTop={1}>
						{item.notes.map((path) => (
							<Text key={path}>
								<Text color={theme.link}>{splitPath(path).title}</Text>
								<Text dimColor>  {splitPath(path).folder}</Text>
							</Text>
						))}
					</Box>
				) : null}
				<Box marginTop={1}>
					<Text dimColor>
						{item.link
							? `link it: [[${splitPath(item.link[1]!).title}]] in ${splitPath(item.link[0]!).title}, and back, under See Also`
							: `link it: not two notes to link (${item.kind})`}
					</Text>
				</Box>
				{item.start_at ? <Text dimColor>pivot: explore from {item.start_at}</Text> : null}
			</Box>
		</Box>
	);
}

/** What stays in the transcript after /explore is closed. */
export function ExploreSummary({data, marks}: {data: ExploreRunResult; marks?: Map<string, Mark>}) {
	const kept = [...(marks?.values() ?? [])].filter((mark) => mark === 'keep').length;
	const dismissed = [...(marks?.values() ?? [])].filter((mark) => mark === 'dismiss').length;
	return (
		<Box flexDirection="column">
			<ExploreHeader data={data} />
			{data.sections.map((section, number) => (
				<Text key={section.lens} wrap="truncate-end">
					{'     '}
					<Text bold>{`${number + 1}. ${section.title}`}</Text>
					<Text dimColor>
						{'  '}
						{section.results.length > 0 ? section.results.map((item) => item.title).join(' · ') : (section.note ?? 'nothing found')}
					</Text>
				</Text>
			))}
			{kept + dismissed > 0 ? (
				<Text dimColor>
					{'     '}this session: kept {kept}, dismissed {dismissed} (vault kept lists what you kept)
				</Text>
			) : null}
		</Box>
	);
}

type Notice = {text: string; ok: boolean};
type Mode = 'list' | 'detail' | 'link';

export type ExploreBrowserProps = {
	api: VaultApi;
	data: ExploreRunResult;
	start: ExploreStart;
	done: (summary: ReactNode) => void;
};

export function ExploreBrowser({api, data: first, start: firstStart, done}: ExploreBrowserProps) {
	const {stdout} = useStdout();
	const terminal = stdout as NodeJS.WriteStream;
	const height = Math.max(8, (terminal.rows && terminal.rows > 0 ? terminal.rows : 24) - 12);
	const [data, setData] = useState(first);
	const [start, setStart] = useState<ExploreStart>(firstStart);
	const [back, setBack] = useState<ExploreStart[]>([]);
	const [selected, setSelected] = useState(0);
	const [mode, setMode] = useState<Mode>('list');
	const [marks, setMarks] = useState(() => new Map<string, Mark>());
	const [notice, setNotice] = useState<Notice | undefined>();
	const [busy, setBusy] = useState(false);
	const [dryRun, setDryRun] = useState<ExploreLinkResult | undefined>();
	const rows = rowsOf(data);
	const items = rows.flatMap((row) => (row.kind === 'item' ? [row.item] : []));
	const current = items[selected];
	const sectionOf = (item: ExploreItem) => data.sections.find((section) => section.results.includes(item))!;
	const markOf = (item: ExploreItem): Mark => (marks.has(item.ref) ? marks.get(item.ref)! : item.mark);

	const fail = (error: unknown) => {
		setNotice({text: error instanceof Error ? error.message : String(error), ok: false});
	};

	const run = async (next: ExploreStart, fresh = false, remember = true) => {
		setBusy(true);
		setNotice({text: `Exploring ${next.label}…`, ok: true});
		try {
			const result = await api.explore({...(next.note === undefined ? {} : {note: next.note}), ...(fresh ? {fresh: true} : {})});
			if (remember && next.note !== start.note) {
				setBack((stack) => [...stack, start]);
			}

			setStart(next);
			setData(result);
			setSelected(0);
			setMarks(new Map());
			setMode('list');
			setNotice(undefined);
		} catch (error) {
			fail(error);
		} finally {
			setBusy(false);
		}
	};

	const mark = async (item: ExploreItem, verdict: 'keep' | 'dismiss' | 'clear') => {
		setBusy(true);
		try {
			const result = await api.mark({verdict, refs: [item.ref]});
			setMarks((previous) => new Map(previous).set(item.ref, verdict === 'clear' ? null : verdict));
			const rates = result.keep_rates.map((r) => `${r.lens}: kept ${r.kept} of ${r.kept + r.dismissed}`).join(' · ');
			const what = verdict === 'keep' ? 'Kept' : verdict === 'dismiss' ? 'Dismissed' : 'Cleared';
			setNotice({text: `${what} ${item.ref} ${item.title} · ${rates}`, ok: true});
		} catch (error) {
			fail(error);
		} finally {
			setBusy(false);
		}
	};

	const link = async (item: ExploreItem, write: boolean) => {
		if (!item.link) {
			setNotice({text: `${item.ref} is not two notes to link (${item.kind})`, ok: false});
			return;
		}

		setBusy(true);
		try {
			const result = await api.linkIt(write ? {ref: item.ref, dry_run: false} : {ref: item.ref});
			setDryRun(result);
			setMode('link');
			if (write && result.status === 'written') {
				setMarks((previous) => new Map(previous).set(item.ref, 'keep'));
				setNotice({text: `Linked ${result.notes.map((path) => splitPath(path).title).join(' ↔ ')}; kept ${item.ref}`, ok: true});
			} else {
				setNotice(undefined);
			}
		} catch (error) {
			fail(error);
		} finally {
			setBusy(false);
		}
	};

	useInput((input, key) => {
		if (busy) {
			return;
		}

		if (mode === 'link') {
			if (key.escape || (dryRun?.status !== 'dry-run' && (input === 'q' || key.return))) {
				setMode('list');
				setDryRun(undefined);
			} else if (input === 'w' && current && dryRun?.status === 'dry-run') {
				void link(current, true);
			}

			return;
		}

		if (key.escape || (mode === 'list' && input === 'q')) {
			if (mode === 'detail') {
				setMode('list');
			} else {
				done(<ExploreSummary data={data} marks={marks} />);
			}

			return;
		}

		if ((key.upArrow || key.downArrow) && items.length > 0) {
			setSelected((index) => (index + (key.upArrow ? items.length - 1 : 1)) % items.length);
			setNotice(undefined);
		} else if (key.return && current && mode === 'list') {
			setMode('detail');
		} else if (input === 'k' && current) {
			void mark(current, 'keep');
		} else if (input === 'd' && current) {
			void mark(current, 'dismiss');
		} else if (input === 'c' && current) {
			void mark(current, 'clear');
		} else if (input === 'l' && current) {
			void link(current, false);
		} else if (input === 'p' && current) {
			const note = pivotTarget(current);
			if (note) {
				void run({note, label: splitPath(note).title});
			}
		} else if (input === 'b' && back.length > 0) {
			const previous = back.at(-1)!;
			setBack((stack) => stack.slice(0, -1));
			void run(previous, false, false);
		} else if (input === 'r') {
			void run(start, false, false);
		} else if (input === 'f') {
			void run(start, true, false);
		}
	});

	// Runs after useInput's own effect: from now on keys arrive here (see SearchView).
	const [ready, setReady] = useState(false);
	useEffect(() => {
		setReady(true);
	}, []);

	const hints: Record<Mode, string> = {
		list: '↑↓ Enter open · k keep · d dismiss · l link · p pivot · b back · r again · f fresh · Esc close',
		detail: 'k keep · d dismiss · c clear · l link it · p pivot · Esc back to the list',
		link: dryRun?.status === 'dry-run' ? 'w write it (both notes) · Esc cancel' : 'Esc back to the list',
	};
	return (
		<Box flexDirection="column">
			{mode === 'list' ? (
				<>
					<ExploreHeader data={data} />
					{windowOf(rows, selected, height).map((row) =>
						row.kind === 'section' ? (
							<SectionRow key={`s${row.number}`} section={row.section} number={row.number} />
						) : row.kind === 'empty' ? (
							<Text key={`e${row.section.lens}`} dimColor wrap="truncate-end">
								{'         '}
								{row.section.note ?? 'nothing found'}
							</Text>
						) : (
							<ItemRow key={row.item.ref} item={row.item} selected={row.index === selected} mark={markOf(row.item)} />
						),
					)}
					<Text dimColor>
						{'\n     '}
						{items.length > 0 ? `${selected + 1} of ${items.length}` : 'no results'}
					</Text>
				</>
			) : mode === 'detail' && current ? (
				<Detail item={current} section={sectionOf(current)} mark={markOf(current)} />
			) : dryRun ? (
				<WriteView result={dryRun} maxLines={12} writeHint="w writes it in both notes; Esc cancels" />
			) : null}
			{notice ? (
				<Box paddingLeft={5}>
					<Text color={notice.ok ? theme.ok : theme.error}>
						{notice.ok ? '✓ ' : '✗ '}
						{notice.text}
					</Text>
				</Box>
			) : null}
			<Box borderStyle="round" borderColor="gray" paddingX={1}>
				<Text dimColor>{ready ? (busy ? 'working…' : hints[mode]) : ' '}</Text>
			</Box>
		</Box>
	);
}
