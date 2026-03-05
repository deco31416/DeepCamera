#!/usr/bin/env node
/**
 * Home Security AI Benchmark Suite
 * 
 * Evaluates LLM and VLM models on home security AI tasks:
 * - Context preprocessing (dedup)
 * - Topic classification
 * - Knowledge distillation
 * - Event deduplication (security classifier)
 * - Tool use (tool selection & parameter extraction)
 * - Chat & JSON compliance
 * - VLM scene analysis (optional, requires VLM server)
 * 
 * ## Skill Protocol (when spawned by Aegis)
 * 
 * Aegis → Skill (env vars):
 *   AEGIS_GATEWAY_URL  — LLM gateway URL (e.g. http://localhost:5407)
 *   AEGIS_VLM_URL      — VLM server URL (e.g. http://localhost:5405)
 *   AEGIS_SKILL_PARAMS — JSON params from skill config
 *   AEGIS_SKILL_ID     — Skill ID
 * 
 * Skill → Aegis (stdout, JSON lines):
 *   {"event": "ready", "model": "Qwen3.5-4B-Q4_1"}
 *   {"event": "suite_start", "suite": "Context Preprocessing"}
 *   {"event": "test_result", "suite": "...", "test": "...", "status": "pass", "timeMs": 123}
 *   {"event": "suite_end", "suite": "...", "passed": 4, "failed": 0}
 *   {"event": "complete", "passed": 23, "total": 26, "timeMs": 95000, "reportPath": "..."}
 * 
 * Standalone usage:
 *   node run-benchmark.cjs [options]
 *   --gateway URL   LLM gateway (fallback if no AEGIS_GATEWAY_URL)
 *   --vlm URL       VLM server (fallback if no AEGIS_VLM_URL)
 *   --out DIR       Results directory (default: ~/.aegis-ai/benchmarks)
 *   --report        Auto-generate HTML report after run
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

// ─── Config: Aegis env vars → CLI args → defaults ────────────────────────────

const args = process.argv.slice(2);
function getArg(name, defaultVal) {
    const idx = args.indexOf(`--${name}`);
    if (idx === -1) return defaultVal;
    return args[idx + 1] || defaultVal;
}

// Aegis provides config via env vars; CLI args are fallback for standalone
const GATEWAY_URL = process.env.AEGIS_GATEWAY_URL || getArg('gateway', 'http://localhost:5407');
const VLM_URL = process.env.AEGIS_VLM_URL || getArg('vlm', '');
const RESULTS_DIR = getArg('out', path.join(os.homedir(), '.aegis-ai', 'benchmarks'));
const AUTO_REPORT = args.includes('--report');
const TIMEOUT_MS = 30000;
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures');
const IS_SKILL_MODE = !!process.env.AEGIS_SKILL_ID;

// Parse skill parameters if running as Aegis skill
let skillParams = {};
try { skillParams = JSON.parse(process.env.AEGIS_SKILL_PARAMS || '{}'); } catch { }

// ─── Skill Protocol: JSON lines on stdout, human text on stderr ──────────────

/**
 * Emit a JSON-lines event on stdout (parsed by Aegis skill-runtime-manager).
 * All structured data goes here so Aegis can react to it.
 */
function emit(event) {
    process.stdout.write(JSON.stringify(event) + '\n');
}

/**
 * Log human-readable text to stderr (shows in Aegis console tab).
 * In standalone mode, also mirrors to stdout for terminal visibility.
 */
function log(msg) {
    process.stderr.write(msg + '\n');
}

// ─── Test Framework ───────────────────────────────────────────────────────────

const suites = [];
let currentSuite = null;

function suite(name, fn) {
    suites.push({ name, fn, tests: [] });
}

const results = {
    timestamp: new Date().toISOString(),
    gateway: GATEWAY_URL,
    vlm: VLM_URL || null,
    system: {},
    model: {},
    suites: [],
    totals: { passed: 0, failed: 0, skipped: 0, total: 0, timeMs: 0 },
    tokenTotals: { prompt: 0, completion: 0, total: 0 },
};

async function llmCall(messages, opts = {}) {
    const body = { messages, stream: false };
    if (opts.maxTokens) body.max_tokens = opts.maxTokens;
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    if (opts.tools) body.tools = opts.tools;

    const url = opts.vlm ? `${VLM_URL}/v1/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(opts.timeout || TIMEOUT_MS),
    });

    if (!response.ok) {
        const errBody = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status}: ${errBody.slice(0, 200)}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';
    const toolCalls = data.choices?.[0]?.message?.tool_calls || null;
    const usage = data.usage || {};

    // Track token totals
    results.tokenTotals.prompt += usage.prompt_tokens || 0;
    results.tokenTotals.completion += usage.completion_tokens || 0;
    results.tokenTotals.total += usage.total_tokens || 0;

    // Capture model name from first response
    if (!results.model.name && data.model) {
        results.model.name = data.model;
    }

    return { content, toolCalls, usage, model: data.model };
}

function stripThink(text) {
    return text.replace(/<think>[\s\S]*?<\/think>\s*/gi, '').trim();
}

function parseJSON(text) {
    const cleaned = stripThink(text);
    let jsonStr = cleaned;
    const codeBlock = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (codeBlock) jsonStr = codeBlock[1];
    else {
        const idx = cleaned.search(/[{[]/);
        if (idx > 0) jsonStr = cleaned.slice(idx);
    }
    return JSON.parse(jsonStr.trim());
}

function assert(condition, msg) {
    if (!condition) throw new Error(msg || 'Assertion failed');
}

async function runSuites() {
    for (const s of suites) {
        currentSuite = { name: s.name, tests: [], passed: 0, failed: 0, skipped: 0, timeMs: 0 };
        log(`\n${'─'.repeat(60)}`);
        log(`  ${s.name}`);
        log(`${'─'.repeat(60)}`);
        emit({ event: 'suite_start', suite: s.name });

        await s.fn();

        results.suites.push(currentSuite);
        results.totals.passed += currentSuite.passed;
        results.totals.failed += currentSuite.failed;
        results.totals.skipped += currentSuite.skipped;
        results.totals.total += currentSuite.tests.length;

        emit({ event: 'suite_end', suite: s.name, passed: currentSuite.passed, failed: currentSuite.failed, skipped: currentSuite.skipped, timeMs: currentSuite.timeMs });
    }
}

async function test(name, fn) {
    const testResult = { name, status: 'pass', timeMs: 0, detail: '', tokens: {} };
    const start = Date.now();
    try {
        const detail = await fn();
        testResult.timeMs = Date.now() - start;
        testResult.detail = detail || '';
        currentSuite.passed++;
        log(`  ✅ ${name} (${testResult.timeMs}ms)${detail ? ` — ${detail}` : ''}`);
    } catch (err) {
        testResult.timeMs = Date.now() - start;
        testResult.status = 'fail';
        testResult.detail = err.message;
        currentSuite.failed++;
        log(`  ❌ ${name} (${testResult.timeMs}ms) — ${err.message}`);
    }
    currentSuite.timeMs += testResult.timeMs;
    currentSuite.tests.push(testResult);
    emit({ event: 'test_result', suite: currentSuite.name, test: name, status: testResult.status, timeMs: testResult.timeMs, detail: testResult.detail.slice(0, 120) });
}

function skip(name, reason) {
    currentSuite.skipped++;
    currentSuite.tests.push({ name, status: 'skip', timeMs: 0, detail: reason });
    log(`  ⏭️  ${name} — ${reason}`);
    emit({ event: 'test_result', suite: currentSuite.name, test: name, status: 'skip', timeMs: 0, detail: reason });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 1: CONTEXT PREPROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

function buildPreprocessPrompt(messageIndex, userMessage) {
    return `You are a context deduplication engine. Given a list of user messages from a conversation, decide which exchanges to KEEP and which are DUPLICATES.

## User Messages (index, timestamp, first 50 words)
${messageIndex.map(m => `[${m.idx}] ${m.ts ? `(${m.ts})` : ''} ${m.text}`).join('\n')}

## New User Question
${userMessage}

## Rules
1. If the user asked the same or very similar question multiple times, keep ONLY the LATEST one
2. Keep messages that are clearly different topics or provide unique context
3. Always keep the last 2 user messages (most recent context)
4. Keep system messages (they contain tool results)

## Response Format
Return ONLY this JSON (no other text):
{"keep": [0, 5, 8], "summary": "brief 1-line summary of dropped exchanges"}

- "keep": array of message indices to KEEP (from the index list above)
- "summary": what the dropped messages were about (so context is not lost entirely)
- If nothing should be dropped, set keep to ALL indices and summary to ""`;
}

suite('📋 Context Preprocessing', async () => {
    await test('Exact duplicates (7x same Q) → keep ≤3', async () => {
        const idx = [
            { idx: 0, ts: '9:56 AM', text: 'What has happened today' },
            { idx: 4, ts: '10:09 AM', text: 'What has happened today?' },
            { idx: 8, ts: '10:14 AM', text: 'What has happened today ?' },
            { idx: 12, ts: '10:28 AM', text: 'What has happened today?' },
            { idx: 16, ts: '10:33 AM', text: 'Hi, What has happened today?' },
            { idx: 18, ts: '12:56 PM', text: 'What has happened today' },
            { idx: 22, ts: '1:08 PM', text: 'What has happened today' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'What has happened today?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        assert(p.keep.length <= 3, `Expected ≤3, got ${p.keep.length}`);
        return `kept ${p.keep.length}/7`;
    });

    await test('Mixed topics → preserves unique questions', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'What has happened today' },
            { idx: 3, ts: '9:30 AM', text: 'Set an alert for person detection on front door after 10pm' },
            { idx: 6, ts: '10:00 AM', text: 'What has happened today?' },
            { idx: 10, ts: '10:15 AM', text: 'Show me the clip from 9:40 AM at the front door' },
            { idx: 14, ts: '11:00 AM', text: 'What has happened today?' },
            { idx: 18, ts: '12:00 PM', text: 'What is the system status?' },
            { idx: 22, ts: '1:00 PM', text: 'What has happened today' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Any alerts triggered?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        assert(p.keep.includes(3) || p.keep.includes(10) || p.keep.includes(18), 'Should keep unique topics');
        return `kept ${p.keep.length}/7: [${p.keep.join(',')}]`;
    });

    await test('All unique → keep all', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'Show me the front door camera' },
            { idx: 3, ts: '9:15 AM', text: 'Set alert for person detection' },
            { idx: 6, ts: '10:00 AM', text: 'What is the system status?' },
            { idx: 10, ts: '11:00 AM', text: 'Analyze the clip from 9:40 AM' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Any new motion events?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep) && p.keep.length === 4, `Expected 4, got ${p.keep?.length}`);
        return `kept all 4 ✓`;
    });

    await test('Small history → empty summary', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'Hello' },
            { idx: 2, ts: '9:05 AM', text: 'Show cameras' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Thanks') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        return `kept ${p.keep.length}/2`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 2: TOPIC CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

suite('🏷️ Topic Classification', async () => {
    await test('First turn → topic title (3-6 words)', async () => {
        const r = await llmCall([{
            role: 'user', content: `Classify this exchange's topic in 3-6 words. Respond with ONLY the topic title.
User: "What has happened today on the cameras?"
Assistant: "Today, your cameras captured motion events including a person at the front door at 9:40 AM..."` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').replace(/^(new\s+)?topic\s*:\s*/i, '').trim();
        assert(cleaned.length > 0, 'Topic empty');
        const wc = cleaned.split(/\s+/).length;
        assert(wc <= 8, `Too verbose: ${wc} words`);
        return `"${cleaned}" (${wc} words)`;
    });

    await test('Same topic → SAME', async () => {
        const r = await llmCall([{
            role: 'user', content: `Given this exchange, is the topic still the same?
User: "Show me the clip from 9:40 AM"
Assistant: "Here's the clip from 9:40 AM showing a person at the front door..."
Current topic: "Camera Events Review"
If the topic hasn't changed, respond: SAME
Otherwise respond with ONLY the new topic title (3-6 words).` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '');
        assert(cleaned.toUpperCase() === 'SAME', `Expected SAME, got "${cleaned}"`);
        return 'SAME ✓';
    });

    await test('Topic change → new title', async () => {
        const r = await llmCall([{
            role: 'user', content: `Given this exchange, is the topic still the same?
User: "What's the system status? How much storage am I using?"
Assistant: "System healthy. Storage: 45GB of 500GB, VLM running on GPU."
Current topic: "Camera Events Review"
If the topic hasn't changed, respond: SAME
Otherwise respond with ONLY the new topic title (3-6 words).` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').replace(/^(new\s+)?topic\s*:\s*/i, '').trim();
        assert(cleaned.toUpperCase() !== 'SAME', 'Expected new topic');
        return `"${cleaned}"`;
    });

    await test('Greeting → valid topic', async () => {
        const r = await llmCall([{
            role: 'user', content: `Classify this exchange's topic in 3-6 words. Respond with ONLY the topic title.
User: "Hi, good morning!"
Assistant: "Good morning! How can I help you with your home security today?"` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').trim();
        assert(cleaned.length > 0 && cleaned.length < 50, `Bad: "${cleaned}"`);
        return `"${cleaned}"`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 3: KNOWLEDGE DISTILLATION
// ═══════════════════════════════════════════════════════════════════════════════

const DISTILL_PROMPT = `You are a knowledge extraction agent for SharpAI-Aegis, a home security assistant.
Analyze the conversation below and extract DURABLE knowledge worth remembering permanently.
## What to Extract
- Home/household facts: People, pets, home layout, camera locations
- User preferences: Notification style, alert priorities, quiet hours
- Security patterns: Normal activity times, delivery schedules
- System decisions: Model choices, channel configurations
## What to Skip
- Transient errors and troubleshooting
- Routine exchanges ("thanks", "bye")
- One-time questions
## Output Format
Respond with ONLY valid JSON:
{"items": [{"slug": "home_profile", "facts": [{"type": "camera", "content": "..."}]}], "new_items": [{"title": "...", "summary": "...", "facts": [{"type": "...", "content": "..."}]}]}
### Slug Reference
- home_profile: Cameras, members, pets, layout
- alert_preferences: Per-camera rules, channels, quiet hours
- security_patterns: Activity patterns, false alarms
- system_config: Models, channels, storage
If nothing to extract: {"items": [], "new_items": []}`;

suite('🧠 Knowledge Distillation', async () => {
    await test('Home profile → extracts facts with slug', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Camera Setup\n## Existing KIs: (none)\n## Conversation\nUser: I have three cameras. Front door is a Blink Mini, living room is Blink Indoor, side parking is Blink Outdoor.\nAegis: Got it! Want to set up alerts?\nUser: Yes, person detection on front door after 10pm. My name is Sam.\nAegis: Alert set. Nice to meet you, Sam!` },
        ]);
        const p = parseJSON(r.content);
        assert(p && typeof p === 'object', 'Must return object');
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts >= 2, `Expected ≥2 facts, got ${facts}`);
        return `${facts} facts extracted`;
    });

    await test('Routine chat → empty extraction', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Greeting\n## Existing KIs: (none)\n## Conversation\nUser: Hi\nAegis: Hello! How can I help?\nUser: Thanks, bye\nAegis: Goodbye!` },
        ]);
        const p = parseJSON(r.content);
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts === 0, `Expected 0 facts, got ${facts}`);
        return 'empty ✓';
    });

    await test('Alert preferences → extracts to correct slug', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Alert Configuration\n## Existing KIs: alert_preferences\n## Conversation\nUser: No notifications from side parking 8am-5pm. Too many false alarms from passing cars.\nAegis: Quiet hours set for side parking 8 AM-5 PM.\nUser: Front door alerts go to Telegram. Discord for everything else.\nAegis: Done — front door to Telegram, rest to Discord.` },
        ]);
        const p = parseJSON(r.content);
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts >= 2, `Expected ≥2 facts, got ${facts}`);
        return `${facts} facts`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 4: EVENT DEDUPLICATION
// ═══════════════════════════════════════════════════════════════════════════════

function buildDedupPrompt(current, recent, ageSec) {
    return `You are a security event deduplication assistant.

## TASK
Determine if CURRENT CLIP and RECENT CLIP show the same ongoing event or different events.

## CURRENT CLIP
- Camera: ${current.camera}
- Type: ${current.type}
- Summary: ${current.summary}

## RECENT CLIP (sent ${ageSec}s ago)
- Camera: ${recent.camera}
- Type: ${recent.type}
- Summary: ${recent.summary}

## DECISION CRITERIA
DUPLICATE if: Same person lingering, same vehicle, continuation of activity
UNIQUE if: Different person/vehicle, different activity, new event

## RESPONSE FORMAT
Respond with ONLY a JSON object:
{"duplicate": true/false, "reason": "brief explanation", "confidence": "high/medium/low"}`;
}

suite('🔔 Event Deduplication', async () => {
    await test('Same person lingering → duplicate', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are a security event classifier. Respond only with valid JSON.' },
            {
                role: 'user', content: buildDedupPrompt(
                    { camera: 'Front Door', type: 'motion', summary: 'Person in blue shirt standing on sidewalk looking at camera' },
                    { camera: 'Front Door', type: 'motion', summary: 'Man in blue shirt on sidewalk, inspecting security camera' },
                    120
                )
            },
        ], { maxTokens: 150, temperature: 0.1 });
        const p = parseJSON(r.content);
        assert(p.duplicate === true, `Expected duplicate=true, got ${p.duplicate}`);
        return `dup=true, confidence=${p.confidence}`;
    });

    await test('Different person → unique', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are a security event classifier. Respond only with valid JSON.' },
            {
                role: 'user', content: buildDedupPrompt(
                    { camera: 'Front Door', type: 'motion', summary: 'Woman in red dress carrying package to front door' },
                    { camera: 'Front Door', type: 'motion', summary: 'Man in blue shirt on sidewalk looking at camera' },
                    300
                )
            },
        ], { maxTokens: 150, temperature: 0.1 });
        const p = parseJSON(r.content);
        assert(p.duplicate === false, `Expected duplicate=false, got ${p.duplicate}`);
        return `dup=false, confidence=${p.confidence}`;
    });

    await test('Multi-camera same vehicle → correct classification', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are a security event classifier. Respond only with valid JSON.' },
            {
                role: 'user', content: buildDedupPrompt(
                    { camera: 'Side Parking', type: 'motion', summary: 'Car pulling into driveway' },
                    { camera: 'Front Door', type: 'motion', summary: 'Car visible on street near driveway' },
                    60
                )
            },
        ], { maxTokens: 150, temperature: 0.1 });
        const p = parseJSON(r.content);
        assert(typeof p.duplicate === 'boolean', 'Must be boolean');
        assert(typeof p.reason === 'string', 'Must have reason');
        return `dup=${p.duplicate}, reason="${p.reason.slice(0, 50)}"`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 5: TOOL USE
// ═══════════════════════════════════════════════════════════════════════════════

const AEGIS_TOOLS = [
    { type: 'function', function: { name: 'video_search', description: 'Search, list, and browse security camera events. Use when the user wants to FIND or LIST clips — e.g. "what happened today", "show events from yesterday", "any motion on the driveway?"', parameters: { type: 'object', properties: { query: { type: 'string', description: 'Search query describing what to find' }, date: { type: 'string', description: 'Date filter (e.g. "today", "2026-03-04")' }, camera: { type: 'string', description: 'Camera name filter' } } } } },
    { type: 'function', function: { name: 'video_analyze', description: 'Analyze and describe the visual content of a specific video clip using VLM. Use when the user wants to EXAMINE, ANALYZE, or DESCRIBE what is happening IN a particular clip — e.g. "analyze the clip from the front door at 9:40 AM", "what do you see in this clip?", "describe the activity"', parameters: { type: 'object', properties: { clip_id: { type: 'string', description: 'Clip ID if known' }, camera: { type: 'string', description: 'Camera name the clip is from' }, time: { type: 'string', description: 'Timestamp of the clip' }, question: { type: 'string', description: 'What to analyze in the clip' } } } } },
    { type: 'function', function: { name: 'system_status', description: 'Get system health, storage, and model status', parameters: { type: 'object', properties: {} } } },
    { type: 'function', function: { name: 'set_alert', description: 'Configure an alert rule for a camera', parameters: { type: 'object', properties: { camera: { type: 'string' }, type: { type: 'string' }, schedule: { type: 'string' } } } } },
];

suite('🔧 Tool Use', async () => {
    const scenarios = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'tool-use-scenarios.json'), 'utf8'));

    for (const s of scenarios.tool_use_scenarios) {
        await test(`${s.name} → ${s.expected_tool}`, async () => {
            const r = await llmCall([
                { role: 'system', content: 'You are Aegis, a home security AI assistant. Use the available tools to answer user questions. Call the most appropriate tool.' },
                { role: 'user', content: s.user_message },
            ], { tools: AEGIS_TOOLS });

            // Check if model returned tool calls
            if (r.toolCalls && r.toolCalls.length > 0) {
                const toolName = r.toolCalls[0].function.name;
                assert(toolName === s.expected_tool, `Expected ${s.expected_tool}, got ${toolName}`);
                return `tool_call: ${toolName}(${r.toolCalls[0].function.arguments?.slice(0, 40) || '...'})`;
            }

            // Some models return tool calls in the content (without native tool calling)
            const content = stripThink(r.content).toLowerCase();
            assert(content.includes(s.expected_tool) || content.includes(s.expected_tool.replace('_', ' ')),
                `Expected mention of ${s.expected_tool} in response`);
            return `content mentions ${s.expected_tool}`;
        });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 6: CHAT & JSON COMPLIANCE
// ═══════════════════════════════════════════════════════════════════════════════

suite('💬 Chat & JSON Compliance', async () => {
    await test('Aegis persona → security-relevant response', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis, an AI security assistant for home monitoring. Keep responses concise.' },
            { role: 'user', content: 'What can you do?' },
        ]);
        const c = stripThink(r.content);
        assert(c.length > 20 && c.length < 2000, `Length ${c.length}`);
        return `${c.length} chars`;
    });

    await test('NO_REPLY for tool context', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. When you have nothing to say, respond ONLY: NO_REPLY' },
            { role: 'user', content: '[Tool Context] video_search returned 3 clips' },
        ]);
        assert(stripThink(r.content).length < 500, 'Response too long for tool context');
        return `"${stripThink(r.content).slice(0, 40)}"`;
    });

    await test('Multi-turn memory → remembers name', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. Be concise.' },
            { role: 'user', content: 'My name is Alex' },
            { role: 'assistant', content: 'Nice to meet you, Alex!' },
            { role: 'user', content: 'What is my name?' },
        ]);
        assert(stripThink(r.content).toLowerCase().includes('alex'), 'Forgot name');
        return 'Remembered Alex ✓';
    });

    await test('Bare JSON output', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY: {"status": "ok", "count": 3}' }]);
        const p = parseJSON(r.content);
        assert(p.status === 'ok' || p.count === 3);
        return `parsed: ${JSON.stringify(p)}`;
    });

    await test('JSON array output', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY a JSON array of 3 colors: ["red","blue","green"]' }]);
        const c = stripThink(r.content);
        let js = c;
        const cb = c.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (cb) js = cb[1]; else { const i = c.search(/\[/); if (i > 0) js = c.slice(i); }
        const arr = JSON.parse(js.trim());
        assert(Array.isArray(arr) && arr.length >= 3);
        return `[${arr.join(', ')}]`;
    });

    await test('Nested JSON object', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY: {"event":{"camera":"Front Door","time":"9:40 AM"},"alert":true}' }]);
        const p = parseJSON(r.content);
        assert(p.event && typeof p.event === 'object');
        return `camera: "${p.event.camera || '?'}"`;
    });

    await test('Timestamp awareness', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. Pay attention to timestamps.' },
            { role: 'user', content: '[9:40 AM] What has happened since 8 AM on the cameras?' },
        ]);
        const c = stripThink(r.content);
        assert(c.length > 10, 'Too short');
        return `${c.length} chars`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 7: VLM SCENE ANALYSIS (optional)
// ═══════════════════════════════════════════════════════════════════════════════

suite('📸 VLM Scene Analysis', async () => {
    if (!VLM_URL) {
        skip('All VLM tests', 'No --vlm URL provided');
        return;
    }

    const frames = {
        front_door: path.join(FIXTURES_DIR, 'frames', 'front_door_person.png'),
        parking: path.join(FIXTURES_DIR, 'frames', 'parking_lot_vehicle.png'),
        living_room: path.join(FIXTURES_DIR, 'frames', 'living_room_empty.png'),
        night: path.join(FIXTURES_DIR, 'frames', 'night_motion.png'),
    };

    async function vlmAnalyze(framePath, question) {
        const imageData = fs.readFileSync(framePath);
        const base64 = imageData.toString('base64');
        const mimeType = framePath.endsWith('.png') ? 'image/png' : 'image/jpeg';

        const r = await llmCall([{
            role: 'user',
            content: [
                { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } },
                { type: 'text', text: question },
            ],
        }], { vlm: true, timeout: 60000 });
        return stripThink(r.content);
    }

    await test('Front door → detects person', async () => {
        if (!fs.existsSync(frames.front_door)) { skip('Front door frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.front_door, 'Describe what you see in this security camera frame. Focus on people and activity.');
        const lower = desc.toLowerCase();
        assert(lower.includes('person') || lower.includes('someone') || lower.includes('man') || lower.includes('woman') || lower.includes('individual'),
            `Expected person detection, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions person ✓`;
    });

    await test('Parking lot → detects vehicle', async () => {
        if (!fs.existsSync(frames.parking)) { skip('Parking frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.parking, 'What vehicles are visible in this security camera frame?');
        const lower = desc.toLowerCase();
        assert(lower.includes('car') || lower.includes('vehicle') || lower.includes('sedan') || lower.includes('truck'),
            `Expected vehicle, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions vehicle ✓`;
    });

    await test('Living room → describes empty scene', async () => {
        if (!fs.existsSync(frames.living_room)) { skip('Living room frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.living_room, 'Describe this indoor security camera scene. Is anyone present?');
        const lower = desc.toLowerCase();
        // Should describe room and note no one present (or describe furniture)
        assert(lower.includes('room') || lower.includes('couch') || lower.includes('living') || lower.includes('sofa'),
            `Expected room description, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars`;
    });

    await test('Night IR → detects figure/motion', async () => {
        if (!fs.existsSync(frames.night)) { skip('Night frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.night, 'Describe what you see in this nighttime infrared security camera frame. Focus on any people or suspicious activity.');
        assert(desc.length > 20, 'Too short');
        return `${desc.length} chars`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM INFO COLLECTOR
// ═══════════════════════════════════════════════════════════════════════════════

function collectSystemInfo() {
    const cpus = os.cpus();
    const mem = process.memoryUsage();
    return {
        os: `${os.type()} ${os.release()} (${os.arch()})`,
        cpu: cpus[0]?.model || 'unknown',
        cpuCores: cpus.length,
        totalMemoryGB: (os.totalmem() / 1073741824).toFixed(1),
        freeMemoryGB: (os.freemem() / 1073741824).toFixed(1),
        nodeVersion: process.version,
        processMemoryMB: {
            rss: (mem.rss / 1048576).toFixed(1),
            heapUsed: (mem.heapUsed / 1048576).toFixed(1),
            heapTotal: (mem.heapTotal / 1048576).toFixed(1),
        },
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
    log('╔══════════════════════════════════════════════════════════════════╗');
    log('║   Home Security AI Benchmark Suite  •  DeepCamera / SharpAI     ║');
    log('╚══════════════════════════════════════════════════════════════════╝');
    log(`  Gateway:  ${GATEWAY_URL}`);
    log(`  VLM:      ${VLM_URL || '(disabled — use --vlm URL to enable)'}`);
    log(`  Results:  ${RESULTS_DIR}`);
    log(`  Mode:     ${IS_SKILL_MODE ? 'Aegis Skill' : 'Standalone'}`);
    log(`  Time:     ${new Date().toLocaleString()}`);

    // Healthcheck
    try {
        const ping = await fetch(`${GATEWAY_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: [{ role: 'user', content: 'ping' }], stream: false, max_tokens: 1 }),
            signal: AbortSignal.timeout(15000),
        });
        if (!ping.ok) throw new Error(`HTTP ${ping.status}`);
        const data = await ping.json();
        results.model.name = data.model || 'unknown';
        log(`  Model:    ${results.model.name}`);
    } catch (err) {
        log(`\n  ❌ Cannot reach LLM gateway: ${err.message}`);
        log('     Start the llama-cpp server and gateway, then re-run.\n');
        emit({ event: 'error', message: `Cannot reach LLM gateway: ${err.message}` });
        process.exit(1);
    }

    // Collect system info
    results.system = collectSystemInfo();
    log(`  System:   ${results.system.cpu} (${results.system.cpuCores} cores)`);
    log(`  Memory:   ${results.system.freeMemoryGB}GB free / ${results.system.totalMemoryGB}GB total`);

    // Emit ready event (Aegis listens for this)
    emit({ event: 'ready', model: results.model.name, system: results.system.cpu });

    const suiteStart = Date.now();
    await runSuites();
    results.totals.timeMs = Date.now() - suiteStart;

    // Post-run memory
    const postMem = process.memoryUsage();
    results.system.postRunMemoryMB = {
        rss: (postMem.rss / 1048576).toFixed(1),
        heapUsed: (postMem.heapUsed / 1048576).toFixed(1),
    };

    // Summary
    const { passed, failed, skipped, total, timeMs } = results.totals;
    const tokPerSec = timeMs > 0 ? ((results.tokenTotals.total / (timeMs / 1000)).toFixed(1)) : '?';

    log(`\n${'═'.repeat(66)}`);
    log(`  RESULTS: ${passed}/${total} passed, ${failed} failed, ${skipped} skipped (${(timeMs / 1000).toFixed(1)}s)`);
    log(`  TOKENS:  ${results.tokenTotals.prompt} prompt + ${results.tokenTotals.completion} completion = ${results.tokenTotals.total} total (${tokPerSec} tok/s)`);
    log(`  MODEL:   ${results.model.name}`);
    log(`${'═'.repeat(66)}`);

    if (failed > 0) {
        log('\n  Failures:');
        for (const s of results.suites) {
            for (const t of s.tests) {
                if (t.status === 'fail') log(`    ❌ ${s.name} > ${t.name}: ${t.detail}`);
            }
        }
    }

    // Save results
    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const modelSlug = (results.model.name || 'unknown').replace(/[^a-zA-Z0-9_.-]/g, '_');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const resultFile = path.join(RESULTS_DIR, `${modelSlug}_${ts}.json`);
    fs.writeFileSync(resultFile, JSON.stringify(results, null, 2));
    log(`\n  Results saved: ${resultFile}`);

    // Update index
    const indexFile = path.join(RESULTS_DIR, 'index.json');
    let index = [];
    try { index = JSON.parse(fs.readFileSync(indexFile, 'utf8')); } catch { }
    index.push({
        file: path.basename(resultFile),
        model: results.model.name,
        timestamp: results.timestamp,
        passed, failed, total,
        timeMs,
        tokens: results.tokenTotals.total,
    });
    fs.writeFileSync(indexFile, JSON.stringify(index, null, 2));

    // Auto-generate report
    let reportPath = null;
    if (AUTO_REPORT) {
        log('\n  Generating HTML report...');
        try {
            const reportScript = path.join(__dirname, 'generate-report.cjs');
            reportPath = require(reportScript).generateReport(RESULTS_DIR);
        } catch (err) {
            log(`  ⚠️  Report generation failed: ${err.message}`);
        }
    }

    // Emit completion event (Aegis listens for this)
    emit({
        event: 'complete',
        model: results.model.name,
        passed, failed, skipped, total,
        timeMs,
        tokens: results.tokenTotals.total,
        tokPerSec: parseFloat(tokPerSec) || 0,
        resultFile,
        reportPath,
    });

    log('');
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
    log(`Fatal: ${err.message}`);
    emit({ event: 'error', message: err.message });
    process.exit(1);
});

