import {RpcError} from '../rpc/client.js';

/**
 * -32003 from the engine: vault-tui was started without --format (say how to fix that), or the format has no
 * block the command needs, e.g. /pipeline with a format without `pipeline:` (name one that has it).
 */
export async function withFormat<T>(call: () => Promise<T>): Promise<T> {
	try {
		return await call();
	} catch (error) {
		if (error instanceof RpcError && error.code === -32_003) {
			if (/has no `\w+:` block/.test(error.message)) {
				throw new Error(`${error.message}: start vault-tui with a format that has one, e.g. --format core (error -32003)`);
			}

			throw new Error(`${error.message.replace(/; start the server with --format.*$/, '')}. Start vault-tui with --format <name>, e.g. --format brain (error -32003)`);
		}

		throw error;
	}
}
