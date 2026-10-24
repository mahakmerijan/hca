const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '..', 'templates/index.html'), 'utf8');
const context = vm.createContext({});
for(const name of ['shouldFallbackToServerStt']){
  const declaration = template.match(new RegExp(`^function ${name}\\([^]*?^}`, 'm'));
  assert.ok(declaration, `${name} should exist`);
  vm.runInContext(declaration[0], context);
}

test('falls back to server transcription when the browser speech service is unavailable', () => {
  for(const err of ['network', 'service-not-allowed', 'language-not-supported']){
    assert.equal(context.shouldFallbackToServerStt(err), true, err);
  }
});

test('does not fall back for user-caused or transient errors', () => {
  for(const err of ['not-allowed', 'no-speech', 'aborted', 'audio-capture']){
    assert.equal(context.shouldFallbackToServerStt(err), false, err);
  }
});

test('hands-free calls use server audio capture without waiting for browser recognition to fail', () => {
  const fn = template.match(/^async function ltStartListening\(\)[^]*?^}/m);
  assert.ok(fn, 'automatic live listening should exist');
  assert.match(fn[0], /startServerVoiceCapture/);
  assert.doesNotMatch(fn[0], /new Recognition/);
  assert.match(template, /fetch\('\/live\/transcribe'/);
});

test('question-form mic buttons also fall back to server transcription', () => {
  const fn = template.match(/^async function startVoiceInput\(qid, btn\)[^]*?^}/m);
  assert.ok(fn, 'startVoiceInput should exist');
  assert.match(fn[0], /voiceUseServerStt\)\{ startServerVoiceInput\(qid, btn, fieldEl\)/);
  assert.match(fn[0], /shouldFallbackToServerStt\(e\.error\)/);
  assert.match(fn[0], /if\(fallBack\) startServerVoiceInput\(qid, btn, fieldEl\)/);
});

test('every SpeechRecognition user has a server fallback', () => {
  const users = [...template.matchAll(/^(?:async )?function (\w+)\([^)]*\)\{[^]*?^}/gm)]
    .filter(m => /new Recognition\(\)/.test(m[0]));
  assert.ok(users.length >= 1);
  for(const m of users) assert.match(m[0], /shouldFallbackToServerStt/, m[1]);
});

test('live-call transcript is multi-line and read-only with no manual send or mic controls', () => {
  const box = template.match(/<(\w+)[^>]*id="lt-msg-input"[^>]*>/);
  assert.ok(box, 'reply box exists');
  assert.equal(box[1], 'textarea');
  assert.match(box[0], /readonly/);
  assert.doesNotMatch(template, /id="lt-(?:mic|send)-btn"/);
  assert.doesNotMatch(box[0], /onkeydown/);
});

test('autoGrowField grows with the text up to a cap and keeps the newest words in view', () => {
  const vm = require('node:vm');
  const ctx = vm.createContext({Math});
  vm.runInContext(template.match(/^function autoGrowField\([^]*?^}/m)[0], ctx);
  const el = {style: {}, scrollHeight: 90, scrollTop: 0};
  ctx.autoGrowField(el, 140);
  assert.equal(el.style.height, '90px');
  assert.equal(el.style.overflowY, 'hidden');
  el.scrollHeight = 300;
  ctx.autoGrowField(el, 140);
  assert.equal(el.style.height, '140px');
  assert.equal(el.style.overflowY, 'auto');
  assert.equal(el.scrollTop, 300);
});
