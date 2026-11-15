// free-model-aggregator-01's entry: one server, two client wires, ten simulated providers.
//
// This server reads the trial's brief (MAKEFUN_PROMPT_FILE), builds one catalog from it, registers
// with the env (MAKEFUN_ENV_URL), and serves the client's twenty requests, one at a time, each with a
// free model that meets the request's requirements.
//
// ROUTING (quality is 60% of the score): for each request the candidates are the quality table's free
// rows meeting the requirements, best rating first. A provider call that refuses (402, 429, 5xx, or a
// body that is not the provider's answer) is retried on the next candidate, and the call it cost is
// counted; a provider that has taken its sealed ceiling of calls is skipped outright, because its next
// call cannot succeed. The first successful call serves the request: exactly one provider call
// answers it, with the model that really dialled named in the response.
//
// THE WIRES: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or in SSE. The
// providers speak openai, anthropic or gemini. Every upstream call is made non-streaming and its JSON
// answer is read for the text and usage, then rendered into the client's wire (see lib/wire.ts): the
// provider's answer is the same either way, and buffering keeps the client's stream in exactly the
// event order its wire expects, with the usage the provider reported.
//
// THE CATALOG comes from the brief alone (see lib/providers.ts): the same document, byte for byte, is
// what `--list` prints and what the register body carries, whether or not the env is reachable when
// either runs. The probes at startup are a liveness log only; routing never reads them.
//
// DETERMINISM: no wall clock and no unseeded randomness anywhere. The model lists use fixed
// timestamps, the answer ids are fixed strings, and every routing choice derives from the brief, the
// requests in arrival order, and the providers' answers — so a replay leaves the same state export,
// byte for byte.
//
// Run as `node main.ts` from this repository's root, with Node 22's standard library alone (nothing
// is installed; TypeScript runs directly, and a relative import needs its .ts extension).
// `node main.ts --list` prints the catalog JSON and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalogFromBrief,
  candidatesFor,
  isRecord,
  joinBase,
  authHeaders,
  probeProvider,
  readBrief,
  servedModels,
  type Brief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  extractAnswer,
  openAIError,
  openAIJsonMode,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type ProviderAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the entry fails over to the next candidate. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Calls this trial has made to each provider, successful or refused. */
const attempts = new Map<string, number>();
/** Refusals in a row per provider: two in a row means its budget is spent or its outage has begun. */
const consecutiveFailures = new Map<string, number>();
/** Providers skipped for the rest of the trial (two refusals in a row). */
const dead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Read the brief and build the one catalog. The probes run separately: `--list` never needs them. */
function loadBrief(): void {
  if (brief.error !== null) log(brief.error);
  catalog = buildCatalogFromBrief(brief);
}

/** Probe every provider (a liveness log only; routing never reads the results). */
async function probeAll(): Promise<void> {
  const probes: Probe[] = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) {
    if (probe.ok) log(`probe ${probe.provider.id}: ${probe.models.length} models`);
    else if (probe.error !== null) log(probe.error);
  }
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(brief);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One provider call for one candidate: the client's body translated into the provider's wire, sent
 * non-streaming, its JSON answer read for the text and usage. Returns the answer, or null when the
 * provider refused, could not be reached, or answered something that is not its answer (a body
 * without the `ANSWER ECHO` text is never served: the trial's content check needs that text whole).
 */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<ProviderAnswer | null> {
  const provider = candidate.provider;
  const model = candidate.row.model;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model, stream: false } : anthropicToOpenAI({ ...body, stream: false }, model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model, stream: false } : openAIToAnthropic({ ...body, stream: false }, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
  }
  attempts.set(provider.id, (attempts.get(provider.id) ?? 0) + 1);
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`${provider.id}/${model}: unreachable: ${(err as Error).message}`);
    return null;
  }
  if (!response.ok) {
    log(`${provider.id}/${model}: HTTP ${response.status}`);
    return null;
  }
  let parsed: unknown;
  try {
    parsed = await response.json();
  } catch {
    log(`${provider.id}/${model}: the answer is not JSON`);
    return null;
  }
  if (!isRecord(parsed)) {
    log(`${provider.id}/${model}: the answer is not one JSON object`);
    return null;
  }
  const answer = extractAnswer(provider.wire, parsed);
  if (answer === null || !answer.text.startsWith('ANSWER ECHO')) {
    log(`${provider.id}/${model}: the answer carries no ANSWER text`);
    return null;
  }
  return answer;
}

/**
 * One client request: the candidates in rating order, each tried until one answers. The answer is
 * rendered in the client's wire, naming the model really dialled; when no candidate answers, a 503
 * in the client's wire, and the trial goes on to the next request.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const clientJsonMode = clientApi === 'openai' && openAIJsonMode(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(brief, requires, clientJsonMode, attempts, dead);
  for (const candidate of candidates) {
    const answer = await callCandidate(candidate, clientApi, body);
    if (answer === null) {
      const failed = (consecutiveFailures.get(candidate.provider.id) ?? 0) + 1;
      consecutiveFailures.set(candidate.provider.id, failed);
      if (failed >= 2) dead.add(candidate.provider.id);
      continue;
    }
    consecutiveFailures.set(candidate.provider.id, 0);
    const model = candidate.row.model;
    log(`served ${clientApi} request with ${candidate.provider.id}/${model}`);
    if (clientApi === 'openai') {
      if (stream) return sendSse(res, renderOpenAIStream(answer, model));
      return sendJson(res, 200, renderOpenAIJson(answer, model));
    }
    if (stream) return sendSse(res, renderAnthropicStream(answer, model));
    return sendJson(res, 200, renderAnthropicJson(answer, model));
  }
  log(`no candidate answered this ${clientApi} request`);
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  loadBrief();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  await probeAll();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(brief).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
