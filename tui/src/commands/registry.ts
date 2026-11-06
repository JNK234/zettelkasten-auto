/**
 * The command registry. A command is one object registered once; dispatch,
 * /help and the "/" suggestions all read from here, so adding a feature means
 * writing a Command and registering it, with no UI changes.
 */
import type {ReactNode} from 'react';
import type {VaultApi} from '../rpc/api.js';
import {parseMentions, type Mention} from './mentions.js';

export type CommandInput = {
	/** Everything after the command name, as typed. */
	args: string;
	/** args with the @-mentions removed. */
	text: string;
	mentions: Mention[];
};

export type CommandContext = {
	api: VaultApi;
	registry: CommandRegistry;
};

export type Command = {
	/** Lower-case, no leading slash: "note" is invoked as /note. */
	name: string;
	summary: string;
	/** Shown in /help, e.g. "/note @note | /note <search>". */
	usage?: string;
	aliases?: readonly string[];
	/** Returns what to show in the transcript. Throwing shows an error entry. */
	run(input: CommandInput, context: CommandContext): Promise<ReactNode>;
};

export type ParsedLine =
	| {kind: 'empty'}
	| {kind: 'text'; text: string}
	| {kind: 'command'; command: Command; input: CommandInput}
	| {kind: 'unknown'; name: string; suggestions: Command[]};

const NAME = /^[a-z][a-z\d-]*$/;

export class CommandRegistry {
	private readonly commands = new Map<string, Command>();
	private readonly lookup = new Map<string, Command>();

	register(...commands: Command[]): this {
		for (const command of commands) {
			const names = [command.name, ...(command.aliases ?? [])];
			for (const name of names) {
				if (!NAME.test(name)) {
					throw new Error(`Invalid command name "${name}" (use lower-case letters, digits, dashes)`);
				}

				if (this.lookup.has(name)) {
					throw new Error(`/${name} is already registered`);
				}
			}

			this.commands.set(command.name, command);
			for (const name of names) {
				this.lookup.set(name, command);
			}
		}

		return this;
	}

	get(name: string): Command | undefined {
		return this.lookup.get(name.toLowerCase());
	}

	list(): Command[] {
		return [...this.commands.values()].sort((a, b) => a.name.localeCompare(b.name));
	}

	/** Commands whose name or alias starts with `prefix` (all commands for ""). */
	suggest(prefix: string): Command[] {
		const wanted = prefix.toLowerCase();
		return this.list().filter((command) =>
			[command.name, ...(command.aliases ?? [])].some((name) => name.startsWith(wanted)),
		);
	}

	parse(line: string): ParsedLine {
		const trimmed = line.trim();
		if (!trimmed) {
			return {kind: 'empty'};
		}

		if (!trimmed.startsWith('/')) {
			return {kind: 'text', text: trimmed};
		}

		const match = /^\/(\S*)\s*([\s\S]*)$/.exec(trimmed);
		const name = match?.[1] ?? '';
		const args = match?.[2] ?? '';
		const command = this.get(name);
		if (!command) {
			return {kind: 'unknown', name, suggestions: this.suggest(name.slice(0, 1))};
		}

		const {mentions, rest} = parseMentions(args);
		return {kind: 'command', command, input: {args, text: rest, mentions}};
	}
}
