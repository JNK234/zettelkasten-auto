/**
 * The prompt: a one-line editor with history, "/" command suggestions and an
 * "@" note picker.
 *
 * Key priority: an open picker or suggestion list takes ↑/↓/Tab/Enter/Esc;
 * otherwise ↑/↓ browse history and Enter submits.
 *
 * Editing state lives in a ref as well as in React state: several key events
 * can arrive before React re-renders (fast typing), and each must see the
 * result of the previous one.
 */
import {Box, Text, useInput} from 'ink';
import {useEffect, useRef, useState} from 'react';
import {activeMention, formatMention} from '../commands/mentions.js';
import type {Command, CommandRegistry} from '../commands/registry.js';
import type {VaultApi} from '../rpc/api.js';
import type {PickedNote} from '../rpc/schema.js';
import type {PromptHistory} from '../state/history.js';
import {splitPath, theme} from './theme.js';

const PICK_LIMIT = 8;
const PICK_DEBOUNCE_MS = 60;

/** recalled: the line came from history; overlays stay closed until it is edited, so ↑/↓ keep browsing. */
type EditorState = {value: string; cursor: number; selected: number; dismissed: string | undefined; recalled: boolean};

type PickerState = {query: string | undefined; results: PickedNote[]; total: number; loading: boolean; error?: string};

function usePicker(api: VaultApi, query: string | undefined): PickerState {
	const [state, setState] = useState<PickerState>({query: undefined, results: [], total: 0, loading: false});
	useEffect(() => {
		if (!query) {
			setState({query, results: [], total: 0, loading: false});
			return;
		}

		let cancelled = false;
		setState((previous) => ({...previous, loading: true}));
		const timer = setTimeout(() => {
			api.pick({query, limit: PICK_LIMIT}).then(
				(result) => {
					if (!cancelled) {
						setState({query, results: result.results, total: result.total, loading: false});
					}
				},
				(error: unknown) => {
					if (!cancelled) {
						setState({query, results: [], total: 0, loading: false, error: error instanceof Error ? error.message : String(error)});
					}
				},
			);
		}, PICK_DEBOUNCE_MS);
		return () => {
			cancelled = true;
			clearTimeout(timer);
		};
	}, [api, query]);
	return state;
}

function InputLine({value, cursor, placeholder, active}: {value: string; cursor: number; placeholder: string; active: boolean}) {
	if (!value) {
		return (
			<Text>
				{active ? <Text inverse> </Text> : null}
				<Text dimColor>{placeholder}</Text>
			</Text>
		);
	}

	return (
		<Text>
			{value.slice(0, cursor)}
			{active ? <Text inverse>{value[cursor] ?? ' '}</Text> : (value[cursor] ?? '')}
			{value.slice(cursor + 1)}
		</Text>
	);
}

/** Matched characters in bold accent. Consecutive characters are grouped into runs (few nodes per row). */
function Highlighted({text, positions}: {text: string; positions: number[]}) {
	const marked = new Set(positions);
	const runs: Array<{text: string; marked: boolean}> = [];
	for (const [index, char] of [...text].entries()) {
		const last = runs.at(-1);
		if (last && last.marked === marked.has(index)) {
			last.text += char;
		} else {
			runs.push({text: char, marked: marked.has(index)});
		}
	}

	return (
		<Text>
			{runs.map((run, index) => (
				// eslint-disable-next-line react/no-array-index-key
				<Text key={index} bold={run.marked} color={run.marked ? theme.accent : undefined}>
					{run.text}
				</Text>
			))}
		</Text>
	);
}

function NotePicker({picker, selected}: {picker: PickerState; selected: number}) {
	if (!picker.query) {
		return <Text dimColor>{'  '}Type to search notes by title or alias…</Text>;
	}

	if (picker.error) {
		return <Text color={theme.error}>{'  '}{picker.error}</Text>;
	}

	if (picker.results.length === 0) {
		return <Text dimColor>{'  '}{picker.loading ? 'Searching…' : `No notes match "${picker.query}"`}</Text>;
	}

	return (
		<Box flexDirection="column">
			{picker.results.map((note, index) => {
				const {folder} = splitPath(note.path);
				const isSelected = index === selected;
				// One line per result, truncated at the terminal edge, so ↑/↓ move row by row.
				return (
					<Text key={note.path} wrap="truncate-end">
						<Text color={theme.accent}>{isSelected ? '❯ ' : '  '}</Text>
						<Text inverse={isSelected}>
							<Highlighted text={note.matched_text} positions={note.positions} />
						</Text>
						{note.matched_field === 'alias' ? <Text dimColor>  alias of {note.title}</Text> : null}
						{folder ? <Text dimColor>  {folder}</Text> : null}
					</Text>
				);
			})}
			{picker.total > picker.results.length ? (
				<Text dimColor>
					{'  '}… {picker.total - picker.results.length} more, keep typing to narrow
				</Text>
			) : null}
		</Box>
	);
}

function CommandSuggestions({commands, selected}: {commands: Command[]; selected: number}) {
	const width = Math.max(...commands.map((command) => command.name.length)) + 4;
	return (
		<Box flexDirection="column">
			{commands.map((command, index) => (
				<Text key={command.name}>
					<Text color={theme.accent}>{index === selected ? '❯ ' : '  '}</Text>
					<Text bold={index === selected}>{`/${command.name}`.padEnd(width)}</Text>
					<Text dimColor>{command.summary}</Text>
				</Text>
			))}
		</Box>
	);
}

/** Which overlay is open, derived from one editor snapshot (used by render and key handler alike). */
function overlays(state: EditorState, registry: CommandRegistry) {
	const mention = activeMention(state.value, state.cursor);
	const mentionKey = mention ? `@${mention.start}` : undefined;
	const pickerOpen = mention !== undefined && state.dismissed !== mentionKey && !state.recalled;
	const typingCommand =
		!state.recalled && state.value.startsWith('/') && !/\s/.test(state.value) && state.cursor === state.value.length;
	const suggestions = typingCommand && state.dismissed !== `/${state.value}` ? registry.suggest(state.value.slice(1)) : [];
	return {mention, mentionKey, pickerOpen, suggestions, suggestionsOpen: !pickerOpen && suggestions.length > 0};
}

export type PromptProps = {
	api: VaultApi;
	registry: CommandRegistry;
	history: PromptHistory;
	busy: boolean;
	onSubmit: (line: string) => void;
	onExit: () => void;
};

export function Prompt({api, registry, history, busy, onSubmit, onExit}: PromptProps) {
	const [state, setState] = useState<EditorState>({value: '', cursor: 0, selected: 0, dismissed: undefined, recalled: false});
	const ref = useRef(state);
	const update = (next: Partial<EditorState>) => {
		ref.current = {...ref.current, ...next};
		setState(ref.current);
	};

	const setLine = (value: string, cursor = value.length, recalled = false) => {
		update({value, cursor, selected: 0, dismissed: undefined, recalled});
	};

	const view = overlays(state, registry);
	const picker = usePicker(api, view.pickerOpen ? view.mention?.query : undefined);

	const submit = (line: string) => {
		if (busy || !line.trim()) {
			return;
		}

		history.add(line);
		setLine('');
		onSubmit(line);
	};

	useInput((input, key) => {
		const {value, cursor, selected} = ref.current;
		const {mention, mentionKey, pickerOpen, suggestions, suggestionsOpen} = overlays(ref.current, registry);

		if (key.ctrl && input === 'd') {
			if (!value) {
				onExit();
			}

			return;
		}

		if (pickerOpen && mention) {
			const count = picker.results.length;
			if (key.upArrow || key.downArrow) {
				if (count > 0) {
					update({selected: (selected + (key.upArrow ? count - 1 : 1)) % count});
				}

				return;
			}

			if (key.tab || key.return) {
				const chosen = picker.results[Math.min(selected, count - 1)];
				if (chosen) {
					const inserted = `${formatMention(chosen.path)} `;
					setLine(value.slice(0, mention.start) + inserted + value.slice(cursor), mention.start + inserted.length);
				}

				return;
			}

			if (key.escape) {
				update({dismissed: mentionKey, selected: 0});
				return;
			}
		} else if (suggestionsOpen) {
			const count = suggestions.length;
			if (key.upArrow || key.downArrow) {
				update({selected: (selected + (key.upArrow ? count - 1 : 1)) % count});
				return;
			}

			const chosen = suggestions[Math.min(selected, count - 1)];
			if (key.tab && chosen) {
				setLine(`/${chosen.name} `);
				return;
			}

			if (key.return && chosen && registry.get(value.slice(1)) === undefined) {
				submit(`/${chosen.name}`);
				return;
			}

			if (key.escape) {
				update({dismissed: `/${value}`, selected: 0});
				return;
			}
		}

		if (key.return) {
			submit(value);
			return;
		}

		if (key.upArrow || key.downArrow) {
			const recalled = key.upArrow ? history.previous(value) : history.next();
			if (recalled !== undefined) {
				setLine(recalled, recalled.length, true);
			}

			return;
		}

		if (key.leftArrow) {
			update({cursor: Math.max(0, cursor - 1)});
		} else if (key.rightArrow) {
			update({cursor: Math.min(value.length, cursor + 1)});
		} else if (key.home || (key.ctrl && input === 'a')) {
			update({cursor: 0});
		} else if (key.end || (key.ctrl && input === 'e')) {
			update({cursor: value.length});
		} else if (key.ctrl && input === 'u') {
			setLine(value.slice(cursor), 0);
		} else if (key.ctrl && input === 'k') {
			setLine(value.slice(0, cursor));
		} else if (key.backspace) {
			if (cursor > 0) {
				update({value: value.slice(0, cursor - 1) + value.slice(cursor), cursor: cursor - 1, selected: 0, dismissed: undefined, recalled: false});
			}
		} else if (key.delete) {
			update({value: value.slice(0, cursor) + value.slice(cursor + 1), selected: 0, dismissed: undefined, recalled: false});
		} else if (input && !key.ctrl && !key.meta && !key.escape && !key.tab) {
			const text = input.replaceAll(/\r?\n/g, ' ');
			update({
				value: value.slice(0, cursor) + text + value.slice(cursor),
				cursor: cursor + text.length,
				selected: 0,
				dismissed: undefined,
				recalled: false,
			});
		}
	});

	return (
		<Box flexDirection="column">
			<Box borderStyle="round" borderColor="gray" paddingX={1}>
				<Text color={busy ? 'gray' : theme.accent}>{'> '}</Text>
				<InputLine value={state.value} cursor={state.cursor} active={!busy} placeholder='Try /help, /stats, /health or /note @…' />
			</Box>
			{view.pickerOpen ? <NotePicker picker={picker} selected={Math.min(state.selected, Math.max(0, picker.results.length - 1))} /> : null}
			{view.suggestionsOpen ? (
				<CommandSuggestions commands={view.suggestions} selected={Math.min(state.selected, view.suggestions.length - 1)} />
			) : null}
		</Box>
	);
}
