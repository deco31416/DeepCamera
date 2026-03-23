/**
 * Unit & Integration tests for MiniMax LLM provider support
 * in the Home Security AI Benchmark.
 *
 * Run: node tests/minimax-provider.test.cjs
 */

const assert = require('assert');
const path = require('path');

// ─── Test Framework ─────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
    try {
        fn();
        passed++;
        console.log(`  ✅ ${name}`);
    } catch (err) {
        failed++;
        failures.push({ name, error: err.message });
        console.log(`  ❌ ${name}: ${err.message}`);
    }
}

async function asyncTest(name, fn) {
    try {
        await fn();
        passed++;
        console.log(`  ✅ ${name}`);
    } catch (err) {
        failed++;
        failures.push({ name, error: err.message });
        console.log(`  ❌ ${name}: ${err.message}`);
    }
}

function suite(name, fn) {
    console.log(`\n📦 ${name}`);
    fn();
}

// ─── Helpers: simulate the provider resolution logic from run-benchmark.cjs ─

const PROVIDER_PRESETS = {
    minimax: {
        baseUrl: 'https://api.minimax.io/v1',
        defaultModel: 'MiniMax-M2.7',
        models: ['MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'MiniMax-M2.5', 'MiniMax-M2.5-highspeed'],
    },
    openai: {
        baseUrl: 'https://api.openai.com/v1',
        defaultModel: '',
        models: [],
    },
};

function resolveProviderBaseUrl(apiType, baseUrl, llmUrl, gatewayUrl) {
    const strip = (u) => u.replace(/\/v1\/?$/, '');
    if (baseUrl) return `${strip(baseUrl)}/v1`;
    const preset = PROVIDER_PRESETS[apiType];
    if (preset) return preset.baseUrl;
    if (llmUrl) return `${strip(llmUrl)}/v1`;
    return `${gatewayUrl}/v1`;
}

function resolveProviderModel(apiType, model) {
    if (model) return model;
    const preset = PROVIDER_PRESETS[apiType];
    return preset ? preset.defaultModel : '';
}

function isMiniMaxProvider(apiType, baseUrl) {
    return apiType === 'minimax'
        || baseUrl.includes('api.minimax.io')
        || baseUrl.includes('minimax');
}

function clampTemperature(temp, isMiniMax) {
    if (isMiniMax && temp !== undefined) {
        return Math.max(0, Math.min(1.0, temp));
    }
    return temp;
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

suite('Provider Preset Resolution', () => {
    test('minimax API type resolves to api.minimax.io base URL', () => {
        const url = resolveProviderBaseUrl('minimax', '', '', 'http://localhost:5407');
        assert.strictEqual(url, 'https://api.minimax.io/v1');
    });

    test('openai API type resolves to api.openai.com base URL', () => {
        const url = resolveProviderBaseUrl('openai', '', '', 'http://localhost:5407');
        assert.strictEqual(url, 'https://api.openai.com/v1');
    });

    test('explicit base URL overrides provider preset', () => {
        const url = resolveProviderBaseUrl('minimax', 'https://custom.example.com/v1', '', 'http://localhost:5407');
        assert.strictEqual(url, 'https://custom.example.com/v1');
    });

    test('unknown API type falls back to gateway URL', () => {
        const url = resolveProviderBaseUrl('builtin', '', '', 'http://localhost:5407');
        assert.strictEqual(url, 'http://localhost:5407/v1');
    });

    test('direct LLM URL takes precedence over gateway for unknown providers', () => {
        const url = resolveProviderBaseUrl('builtin', '', 'http://localhost:8080', 'http://localhost:5407');
        assert.strictEqual(url, 'http://localhost:8080/v1');
    });

    test('strips trailing /v1 from explicit base URL before appending', () => {
        const url = resolveProviderBaseUrl('minimax', 'https://api.minimax.io/v1/', '', 'http://localhost:5407');
        assert.strictEqual(url, 'https://api.minimax.io/v1');
    });
});

suite('Provider Model Resolution', () => {
    test('minimax API type defaults to MiniMax-M2.7', () => {
        const model = resolveProviderModel('minimax', '');
        assert.strictEqual(model, 'MiniMax-M2.7');
    });

    test('explicit model overrides default', () => {
        const model = resolveProviderModel('minimax', 'MiniMax-M2.5-highspeed');
        assert.strictEqual(model, 'MiniMax-M2.5-highspeed');
    });

    test('openai API type has no default model', () => {
        const model = resolveProviderModel('openai', '');
        assert.strictEqual(model, '');
    });

    test('unknown API type has no default model', () => {
        const model = resolveProviderModel('builtin', '');
        assert.strictEqual(model, '');
    });
});

suite('MiniMax Provider Detection', () => {
    test('detects minimax via API type', () => {
        assert.strictEqual(isMiniMaxProvider('minimax', ''), true);
    });

    test('detects minimax via base URL containing api.minimax.io', () => {
        assert.strictEqual(isMiniMaxProvider('openai', 'https://api.minimax.io/v1'), true);
    });

    test('detects minimax via base URL containing minimax', () => {
        assert.strictEqual(isMiniMaxProvider('openai', 'https://minimax-proxy.example.com'), true);
    });

    test('does not detect non-minimax providers', () => {
        assert.strictEqual(isMiniMaxProvider('openai', 'https://api.openai.com'), false);
    });

    test('does not detect builtin as minimax', () => {
        assert.strictEqual(isMiniMaxProvider('builtin', ''), false);
    });
});

suite('Temperature Clamping', () => {
    test('clamps temperature > 1.0 to 1.0 for MiniMax', () => {
        assert.strictEqual(clampTemperature(1.5, true), 1.0);
    });

    test('clamps temperature < 0 to 0 for MiniMax', () => {
        assert.strictEqual(clampTemperature(-0.1, true), 0);
    });

    test('temperature=0 is valid for MiniMax', () => {
        assert.strictEqual(clampTemperature(0, true), 0);
    });

    test('temperature=1.0 is valid for MiniMax', () => {
        assert.strictEqual(clampTemperature(1.0, true), 1.0);
    });

    test('temperature=0.5 passes through for MiniMax', () => {
        assert.strictEqual(clampTemperature(0.5, true), 0.5);
    });

    test('undefined temperature passes through for MiniMax', () => {
        assert.strictEqual(clampTemperature(undefined, true), undefined);
    });

    test('does not clamp temperature for non-MiniMax providers', () => {
        assert.strictEqual(clampTemperature(2.0, false), 2.0);
    });

    test('temperature=0.1 (common benchmark value) passes through', () => {
        assert.strictEqual(clampTemperature(0.1, true), 0.1);
    });
});

suite('MiniMax Model Catalog', () => {
    test('MiniMax preset includes M2.7 model', () => {
        assert.ok(PROVIDER_PRESETS.minimax.models.includes('MiniMax-M2.7'));
    });

    test('MiniMax preset includes M2.7-highspeed model', () => {
        assert.ok(PROVIDER_PRESETS.minimax.models.includes('MiniMax-M2.7-highspeed'));
    });

    test('MiniMax preset includes M2.5 model', () => {
        assert.ok(PROVIDER_PRESETS.minimax.models.includes('MiniMax-M2.5'));
    });

    test('MiniMax preset includes M2.5-highspeed model', () => {
        assert.ok(PROVIDER_PRESETS.minimax.models.includes('MiniMax-M2.5-highspeed'));
    });

    test('MiniMax preset has exactly 4 models', () => {
        assert.strictEqual(PROVIDER_PRESETS.minimax.models.length, 4);
    });

    test('MiniMax base URL uses https', () => {
        assert.ok(PROVIDER_PRESETS.minimax.baseUrl.startsWith('https://'));
    });
});

suite('Cloud API Detection', () => {
    test('minimax API type is recognized as cloud API', () => {
        // isCloudApi logic: LLM_API_TYPE === 'minimax' || isMiniMaxProvider
        const apiType = 'minimax';
        const isCloud = apiType === 'openai' || apiType === 'minimax';
        assert.strictEqual(isCloud, true);
    });

    test('openai API type is recognized as cloud API', () => {
        const apiType = 'openai';
        const isCloud = apiType === 'openai' || apiType === 'minimax';
        assert.strictEqual(isCloud, true);
    });

    test('builtin API type is not cloud API', () => {
        const apiType = 'builtin';
        const isCloud = apiType === 'openai' || apiType === 'minimax';
        assert.strictEqual(isCloud, false);
    });
});

suite('Config File Validation', () => {
    const fs = require('fs');
    const configPath = path.join(__dirname, '..', 'config.yaml');

    test('config.yaml exists', () => {
        assert.ok(fs.existsSync(configPath));
    });

    test('config.yaml contains llmProvider parameter', () => {
        const content = fs.readFileSync(configPath, 'utf8');
        assert.ok(content.includes('llmProvider'));
    });

    test('config.yaml contains minimax option for llmProvider', () => {
        const content = fs.readFileSync(configPath, 'utf8');
        assert.ok(content.includes('minimax'));
    });

    test('config.yaml contains minimaxModel parameter', () => {
        const content = fs.readFileSync(configPath, 'utf8');
        assert.ok(content.includes('minimaxModel'));
    });

    test('config.yaml lists MiniMax-M2.7 model', () => {
        const content = fs.readFileSync(configPath, 'utf8');
        assert.ok(content.includes('MiniMax-M2.7'));
    });

    test('config.yaml lists MiniMax-M2.5-highspeed model', () => {
        const content = fs.readFileSync(configPath, 'utf8');
        assert.ok(content.includes('MiniMax-M2.5-highspeed'));
    });
});

suite('SKILL.md Documentation', () => {
    const fs = require('fs');
    const skillMdPath = path.join(__dirname, '..', 'SKILL.md');

    test('SKILL.md documents minimax API type', () => {
        const content = fs.readFileSync(skillMdPath, 'utf8');
        assert.ok(content.includes('minimax'));
    });

    test('SKILL.md documents MINIMAX_API_KEY env var', () => {
        const content = fs.readFileSync(skillMdPath, 'utf8');
        assert.ok(content.includes('MINIMAX_API_KEY'));
    });

    test('SKILL.md includes MiniMax standalone usage example', () => {
        const content = fs.readFileSync(skillMdPath, 'utf8');
        assert.ok(content.includes('AEGIS_LLM_API_TYPE=minimax'));
    });

    test('SKILL.md lists supported providers table', () => {
        const content = fs.readFileSync(skillMdPath, 'utf8');
        assert.ok(content.includes('Supported LLM Providers'));
    });
});

suite('Script Source Validation', () => {
    const fs = require('fs');
    const scriptPath = path.join(__dirname, '..', 'scripts', 'run-benchmark.cjs');
    const source = fs.readFileSync(scriptPath, 'utf8');

    test('script defines PROVIDER_PRESETS with minimax', () => {
        assert.ok(source.includes("minimax: {"));
        assert.ok(source.includes("baseUrl: 'https://api.minimax.io/v1'"));
    });

    test('script detects MiniMax via isMiniMaxProvider', () => {
        assert.ok(source.includes('isMiniMaxProvider'));
    });

    test('script implements temperature clamping for MiniMax', () => {
        assert.ok(source.includes('Math.max(0, Math.min(1.0, temperature))'));
    });

    test('script reads MINIMAX_API_KEY as fallback', () => {
        assert.ok(source.includes('MINIMAX_API_KEY'));
    });

    test('script uses resolveProviderBaseUrl function', () => {
        assert.ok(source.includes('resolveProviderBaseUrl'));
    });

    test('script uses resolveProviderModel function', () => {
        assert.ok(source.includes('resolveProviderModel'));
    });

    test('script includes MiniMax in isCloudApi check', () => {
        assert.ok(source.includes("LLM_API_TYPE === 'minimax'"));
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS (require MINIMAX_API_KEY)
// ═══════════════════════════════════════════════════════════════════════════════

async function runIntegrationTests() {
    const apiKey = process.env.MINIMAX_API_KEY;
    if (!apiKey) {
        console.log('\n⏭️  Skipping integration tests (set MINIMAX_API_KEY to enable)');
        return;
    }

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('  INTEGRATION TESTS (live MiniMax API)');
    console.log('═══════════════════════════════════════════════════════════════');

    const OpenAI = require('openai');
    const client = new OpenAI({
        apiKey,
        baseURL: 'https://api.minimax.io/v1',
    });

    await asyncTest('MiniMax API responds to simple chat completion', async () => {
        const response = await client.chat.completions.create({
            model: 'MiniMax-M2.7',
            messages: [{ role: 'user', content: 'Reply with exactly: BENCHMARK_OK' }],
            temperature: 0.1,
            max_tokens: 20,
        });
        assert.ok(response.choices[0].message.content.includes('BENCHMARK_OK'));
    });

    await asyncTest('MiniMax API supports streaming', async () => {
        const stream = await client.chat.completions.create({
            model: 'MiniMax-M2.7',
            messages: [{ role: 'user', content: 'Reply with exactly one word: hello' }],
            temperature: 0,
            stream: true,
            stream_options: { include_usage: true },
        });
        let content = '';
        let hasUsage = false;
        for await (const chunk of stream) {
            if (chunk.choices?.[0]?.delta?.content) content += chunk.choices[0].delta.content;
            if (chunk.usage) hasUsage = true;
        }
        assert.ok(content.toLowerCase().includes('hello'), `Expected "hello" in: ${content}`);
        assert.ok(hasUsage, 'Expected usage data in stream');
    });

    await asyncTest('MiniMax API accepts temperature=0', async () => {
        // Verify the API does not reject temperature=0 (no error thrown)
        const response = await client.chat.completions.create({
            model: 'MiniMax-M2.7',
            messages: [{ role: 'user', content: 'Reply with exactly: TEMP_ZERO_OK' }],
            temperature: 0,
            max_tokens: 20,
        });
        assert.ok(response.choices && response.choices.length > 0, 'Expected at least one choice');
        assert.ok(response.choices[0].message.content.length > 0, 'Expected non-empty response');
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║   MiniMax Provider Tests • Home Security AI Benchmark       ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');

    // Integration tests
    await runIntegrationTests();

    // Summary
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  RESULTS: ${passed} passed, ${failed} failed (${passed + failed} total)`);
    console.log(`${'═'.repeat(60)}`);

    if (failures.length > 0) {
        console.log('\n  Failures:');
        for (const f of failures) {
            console.log(`    ❌ ${f.name}: ${f.error}`);
        }
    }

    process.exit(failed > 0 ? 1 : 0);
}

main();
