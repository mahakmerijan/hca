const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '..', 'templates/index.html'), 'utf8');
const context = vm.createContext({Float32Array, ArrayBuffer, DataView, Uint8Array, Math});
for(const name of ['audioRms', 'concatFloat32', 'createSpeechSegmenter', 'downsampleAudio', 'encodeWav']){
  const declaration = template.match(new RegExp(`^function ${name}\\([^]*?^}`, 'm'));
  assert.ok(declaration, `${name} should exist`);
  vm.runInContext(declaration[0], context);
}

const RATE = 1000; // 1 sample per ms keeps the arithmetic readable
const chunk = (ms, amp) => new Float32Array(ms).fill(amp);
const speech = ms => chunk(ms, 0.2);
const silence = ms => chunk(ms, 0);

function seg(opts = {}){
  return context.createSpeechSegmenter({sampleRate: RATE, threshold: 0.05, silenceMs: 500, maxMs: 4000, prerollMs: 200, minSpeechMs: 200, ...opts});
}

test('emits a phrase as soon as the speaker pauses', () => {
  const s = seg();
  assert.deepEqual([...s.push(silence(300))], []);
  assert.equal(s.push(speech(1000)).length, 0);
  assert.equal(s.push(silence(400)).length, 0, 'short gaps between words do not split');
  const out = s.push(silence(100));
  assert.equal(out.length, 1);
  // preroll (200) + speech (1000) + trailing silence (500)
  assert.equal(out[0].length, 1700);
});

test('keeps a short preroll so the first syllable is not clipped', () => {
  const s = seg();
  s.push(silence(1000));
  s.push(speech(500));
  const [phrase] = s.push(silence(500));
  assert.equal(phrase[0], 0);
  assert.equal(phrase[200], Math.fround(0.2));
  assert.equal(phrase[199], 0);
});

test('splits long continuous speech at the max phrase length', () => {
  const s = seg();
  const out = s.push(speech(4000));
  assert.equal(out.length, 1);
  assert.ok(out[0].length <= 4000);
});

test('ignores clicks and blips shorter than the minimum speech length', () => {
  const s = seg();
  s.push(speech(50));
  assert.equal(s.push(silence(600)).length, 0);
});

test('flush returns trailing speech when the user stops mid-phrase', () => {
  const s = seg();
  s.push(speech(800));
  const tail = s.flush();
  assert.ok(tail && tail.length >= 800);
  assert.equal(s.flush(), null);
});

test('downsamples to 16 kHz', () => {
  const out = context.downsampleAudio(new Float32Array(48000).fill(0.5), 48000, 16000);
  assert.equal(out.length, 16000);
  assert.ok(Math.abs(out[100] - 0.5) < 1e-6);
  const same = new Float32Array(10);
  assert.equal(context.downsampleAudio(same, 16000, 16000), same);
});

test('encodes a valid 16-bit mono PCM WAV', () => {
  const buf = context.encodeWav(new Float32Array([0, 1, -1, 0.5]), 16000);
  const v = new DataView(buf);
  const str = (o, n) => String.fromCharCode(...new Uint8Array(buf, o, n));
  assert.equal(str(0, 4), 'RIFF');
  assert.equal(str(8, 4), 'WAVE');
  assert.equal(v.getUint16(22, true), 1, 'mono');
  assert.equal(v.getUint32(24, true), 16000);
  assert.equal(v.getUint16(34, true), 16);
  assert.equal(v.getUint32(40, true), 8);
  assert.equal(v.getInt16(46, true), 32767);
  assert.equal(v.getInt16(48, true), -32768);
});

test('server capture streams phrases instead of one recording', () => {
  const fn = template.match(/^async function startServerVoiceCapture\([^]*?^}/m)[0];
  assert.match(fn, /createSpeechSegmenter/);
  assert.doesNotMatch(fn, /MediaRecorder/);
});

test('current() exposes the in-progress phrase for interim transcription', () => {
  const s = seg();
  assert.equal(s.current(), null);
  s.push(speech(100));
  assert.equal(s.current(), null, 'too short to transcribe yet');
  s.push(speech(400));
  assert.equal(s.current().length, 500);
  s.push(silence(500));
  assert.equal(s.current(), null, 'phrase was cut and handed off');
});

function loadAssembler(){
  const ctx = vm.createContext({Map});
  vm.runInContext(template.match(/^function createTranscriptAssembler\([^]*?^}/m)[0], ctx);
  return ctx.createTranscriptAssembler();
}

test('interim text is shown live and replaced by the final text', () => {
  const a = loadAssembler();
  assert.equal(a.setInterim(0, 'she is', 100), true);
  assert.equal(a.text(), 'she is');
  a.setInterim(0, 'she is a senior', 200);
  assert.equal(a.text(), 'she is a senior');
  a.setFinal(0, "she's a senior recruiter");
  assert.equal(a.text(), "she's a senior recruiter");
});

test('later phrases stay in order while earlier finals are still pending', () => {
  const a = loadAssembler();
  a.setInterim(0, 'first part', 100);
  a.setInterim(1, 'second', 100);
  assert.equal(a.text(), 'first part second');
  a.setFinal(1, 'second phrase');
  assert.equal(a.text(), 'first part second phrase');
  a.setFinal(0, 'first phrase');
  assert.equal(a.text(), 'first phrase second phrase');
});

test('stale or late interim results never overwrite newer text', () => {
  const a = loadAssembler();
  a.setInterim(0, 'longer newer text', 300);
  assert.equal(a.setInterim(0, 'older', 200), false);
  assert.equal(a.text(), 'longer newer text');
  a.setFinal(0, 'final');
  assert.equal(a.setInterim(0, 'late interim', 400), false);
  assert.equal(a.text(), 'final');
});

test('an empty or failed final keeps the words already shown for that phrase', () => {
  const a = loadAssembler();
  a.setInterim(0, 'I led the payments team', 100);
  a.setFinal(0, '');
  assert.equal(a.text(), 'I led the payments team');
  assert.equal(a.hasText(), true, 'kept interim text counts as captured speech');
  assert.equal(a.setInterim(0, 'late', 400), false, 'phrase is locked after its final');
});

test('empty finals with no interim add nothing and hasText reflects captured speech', () => {
  const a = loadAssembler();
  a.setFinal(0, '');
  assert.equal(a.text(), '');
  assert.equal(a.hasText(), false);
  a.setFinal(1, 'hello');
  assert.equal(a.hasText(), true);
});

test('long continuous speech is cut at the quietest moment, not mid-word', () => {
  const s = seg({maxMs: 4000});
  s.push(speech(3000));
  s.push(chunk(100, 0.06)); // a soft dip between words (still above threshold)
  const out = s.push(speech(900));
  assert.equal(out.length, 1);
  assert.equal(out[0].length, 3100, 'cut right after the quiet dip');
  const cur = s.current();
  assert.ok(cur && cur.length === 900, 'speech after the cut carries into the next phrase');
});

test('server capture streams interim updates while the user is speaking', () => {
  const fn = template.match(/^async function startServerVoiceCapture\([^]*?^}/m)[0];
  assert.match(fn, /segmenter\.current\(\)/);
  assert.match(fn, /setInterim\(/);
  assert.match(fn, /onUpdate\(/);
});
