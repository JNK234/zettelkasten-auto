/**
 * Opening notes in Obsidian through its URI scheme:
 * obsidian://open?vault=<vault folder name>&file=<vault-relative path>.
 *
 * The URL is handed to the system opener: $VAULT_TUI_OPENER if set (any program
 * that takes the URL as its one argument, e.g. for a different browser, or a
 * script in tests), otherwise `open` (macOS), `xdg-open` (Linux) or
 * `cmd /c start` (Windows).
 */
import {spawn} from 'node:child_process';

export type UrlOpener = (url: string) => Promise<void>;

export function obsidianUrl(vaultName: string, path: string): string {
	return `obsidian://open?vault=${encodeURIComponent(vaultName)}&file=${encodeURIComponent(path)}`;
}

function openerCommand(url: string): [string, string[]] {
	const custom = process.env.VAULT_TUI_OPENER;
	if (custom) {
		return [custom, [url]];
	}

	if (process.platform === 'darwin') {
		return ['open', [url]];
	}

	if (process.platform === 'win32') {
		return ['cmd', ['/c', 'start', '""', url]];
	}

	return ['xdg-open', [url]];
}

/** Resolves once the opener has started (it runs on, detached); rejects if it cannot be started. */
export const openUrl: UrlOpener = async (url) =>
	new Promise((resolve, reject) => {
		const [command, args] = openerCommand(url);
		const child = spawn(command, args, {detached: true, stdio: 'ignore'});
		child.once('error', (error) => {
			reject(new Error(`could not run ${command}: ${error.message}`));
		});
		child.once('spawn', () => {
			child.unref();
			resolve();
		});
	});
