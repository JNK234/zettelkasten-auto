/**
 * JSON-RPC 2.0 client over a pair of streams, newline-delimited JSON framing
 * (one message per line), matching `python -m vaultengine serve`.
 */
import {EventEmitter} from 'node:events';
import {createInterface} from 'node:readline';
import type {Readable, Writable} from 'node:stream';

export class RpcError extends Error {
	constructor(
		readonly code: number,
		message: string,
		readonly data?: unknown,
	) {
		super(message);
		this.name = 'RpcError';
	}
}

/** The connection to the server is gone (process exited, pipe closed). */
export class ConnectionClosedError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'ConnectionClosedError';
	}
}

type Pending = {
	method: string;
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
};

type ResponseMessage = {
	id?: unknown;
	result?: unknown;
	error?: {code: number; message: string; data?: unknown};
};

export type ClientEvents = {
	close: [error: Error];
	'protocol-error': [line: string];
};

export class JsonRpcClient {
	readonly events = new EventEmitter<ClientEvents>();
	private nextId = 1;
	private readonly pending = new Map<number, Pending>();
	private closedWith: Error | undefined;

	/**
	 * @param incoming the server's stdout
	 * @param outgoing the server's stdin
	 * @param options.failOnEnd fail when `incoming` ends (default true). A process
	 *   owner can pass false and call fail() itself with a better reason.
	 */
	constructor(
		incoming: Readable,
		private readonly outgoing: Writable,
		options: {failOnEnd?: boolean} = {},
	) {
		const lines = createInterface({input: incoming, crlfDelay: Number.POSITIVE_INFINITY});
		lines.on('line', (line) => {
			this.handleLine(line);
		});
		if (options.failOnEnd ?? true) {
			lines.on('close', () => {
				this.fail(new ConnectionClosedError('The vault server closed the connection.'));
			});
		}
	}

	get isClosed(): boolean {
		return this.closedWith !== undefined;
	}

	async request<T>(method: string, params?: object): Promise<T> {
		if (this.closedWith) {
			throw this.closedWith;
		}

		const id = this.nextId++;
		const message = {jsonrpc: '2.0', id, method, ...(params === undefined ? {} : {params})};
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, {method, resolve: resolve as (value: unknown) => void, reject});
			this.outgoing.write(`${JSON.stringify(message)}\n`);
		});
	}

	/** Reject everything in flight and refuse new requests. Idempotent. */
	fail(error: Error): void {
		if (this.closedWith) {
			return;
		}

		this.closedWith = error;
		for (const pending of this.pending.values()) {
			pending.reject(error);
		}

		this.pending.clear();
		this.events.emit('close', error);
	}

	close(): void {
		this.outgoing.end();
	}

	private handleLine(line: string): void {
		if (!line.trim()) {
			return;
		}

		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			this.events.emit('protocol-error', line);
			return;
		}

		for (const message of (Array.isArray(parsed) ? parsed : [parsed]) as ResponseMessage[]) {
			const pending = typeof message.id === 'number' ? this.pending.get(message.id) : undefined;
			if (!pending) {
				this.events.emit('protocol-error', line);
				continue;
			}

			this.pending.delete(message.id as number);
			if (message.error) {
				pending.reject(new RpcError(message.error.code, message.error.message, message.error.data));
			} else {
				pending.resolve(message.result);
			}
		}
	}
}
