import {RpcError} from '../rpc/client.js';

/** -32003 from the engine means vault-tui was started without --format: say how to fix that. */
export async function withFormat<T>(call: () => Promise<T>): Promise<T> {
	try {
		return await call();
	} catch (error) {
		if (error instanceof RpcError && error.code === -32_003) {
			throw new Error(`${error.message.replace(/; start the server with --format.*$/, '')}. Start vault-tui with --format <name>, e.g. --format brain (error -32003)`);
		}

		throw error;
	}
}
