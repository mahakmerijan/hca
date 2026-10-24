const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '..', 'templates/index.html'), 'utf8');
function load(ctx, name){
  const match = template.match(new RegExp(`^(?:async )?function ${name}\\([^]*?^}`, 'm'));
  assert.ok(match, `${name} should exist`);
  vm.runInContext(match[0], ctx);
}
const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred(){
  let resolve, reject;
  const promise = new Promise((a,b) => { resolve=a; reject=b; });
  return {promise, resolve, reject};
}

function captureHarness(){
  let processor;
  const requests=[], turns=[], updates=[], errors=[], states=[];
  const track = {readyState:'live', stop(){ throw new Error('Shared call track must not be stopped'); }};
  const node = () => ({connect(){}, disconnect(){}});
  class AudioContext {
    sampleRate=1000;
    state='running';
    destination={};
    createMediaStreamSource(){ return node(); }
    createScriptProcessor(){ processor={...node(), onaudioprocess:null}; return processor; }
    createGain(){ return {...node(), gain:{value:0}}; }
    close(){ return Promise.resolve(); }
    resume(){ return Promise.resolve(); }
  }
  const ctx=vm.createContext({
    Float32Array, ArrayBuffer, DataView, Uint8Array, Map, Math, console, AbortController,
    window:{AudioContext}, navigator:{mediaDevices:{}},
    MediaStream:class {}, transcribeWav(wav,signal){
      const d={...deferred(),signal}; requests.push(d); return d.promise;
    }
  });
  for(const name of ['audioRms','concatFloat32','createSpeechSegmenter','createTranscriptAssembler',
    'downsampleAudio','encodeWav','startServerVoiceCapture']) load(ctx,name);
  return {
    requests, turns, updates, errors, states,
    start:(options={}) => ctx.startServerVoiceCapture({
      stream:{getAudioTracks:()=>[track]}, silenceMs:1500,
      onTurn:text=>turns.push(text), onUpdate:text=>updates.push(text),
      onError:message=>errors.push(message), onStateChange:state=>states.push(state),
      ...options,
    }),
    push(ms, amplitude){
      assert.ok(processor.onaudioprocess, 'capture should still be listening');
      processor.onaudioprocess({inputBuffer:{getChannelData:()=>new Float32Array(ms).fill(amplitude)}});
    }
  };
}

test('1500 ms of silence ends one turn, waits for transcription, then emits it exactly once', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(300,0.2);
  h.push(1499,0);
  assert.equal(capture.active,true);
  assert.equal(h.turns.length,0);
  h.push(1,0);
  assert.equal(capture.active,false);
  assert.ok(h.states.includes('transcribing'));
  assert.equal(h.turns.length,0,'network result must be awaited');
  for(const d of h.requests) d.resolve('My whole answer');
  await tick();
  assert.deepEqual(h.turns,['My whole answer']);
});

test('a one-second pause followed by more speech resets the 1500 ms endpoint', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(300,0.2); h.push(1000,0); h.push(300,0.2); h.push(1499,0);
  assert.equal(capture.active,true);
  h.push(1,0);
  for(const d of h.requests) d.resolve('Both halves');
  await tick();
  assert.deepEqual(h.turns,['Both halves']);
});

test('silence and short noises do not trigger automatic empty responses', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(2000,0); h.push(50,0.2); h.push(1500,0);
  assert.equal(capture.active,true);
  assert.equal(h.requests.length,0);
  assert.deepEqual(h.turns,[]);
  capture.cancel();
});

test('a long continuous answer is not submitted at the 12-second phrase limit', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(12000,0.2);
  for(const d of h.requests) d.resolve('First long part');
  await tick();
  assert.equal(capture.active,true);
  assert.deepEqual(h.turns,[]);
  h.push(300,0.2); h.push(1500,0);
  for(const d of h.requests.slice(1)) d.resolve('Last part');
  await tick();
  assert.deepEqual(h.turns,['First long part Last part']);
});

test('a pause exactly after the 12-second cut still submits the full answer', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(12000,0.2);
  h.push(1499,0);
  assert.equal(capture.active,true);
  h.push(1,0);
  assert.equal(capture.active,false);
  for(const d of h.requests) d.resolve('All twelve seconds');
  await tick();
  assert.deepEqual(h.turns,['All twelve seconds']);
});

test('late interim speech is preserved when the final is empty', async () => {
  const h=captureHarness();
  await h.start();
  h.push(400,0.2); h.push(1500,0);
  assert.equal(h.requests.length,2);
  h.requests[1].resolve('');
  await tick();
  assert.deepEqual(h.turns,[], 'must wait for outstanding interim');
  h.requests[0].resolve('Café 東京 <script>literal</script>');
  await tick();
  assert.deepEqual(h.turns,['Café 東京 <script>literal</script>']);
});

test('transcription failure is reported instead of generating an empty twin response', async () => {
  const h=captureHarness();
  await h.start();
  h.push(300,0.2); h.push(1500,0);
  for(const d of h.requests) d.reject(new Error('Service unavailable'));
  await tick();
  assert.ok(h.errors.some(message=>message.includes('Service unavailable')));
  assert.deepEqual(h.turns,['']);
});

test('cancellation drops in-flight transcription and all subsequent updates', async () => {
  const h=captureHarness();
  const capture=await h.start();
  h.push(400,0.2);
  capture.cancel();
  assert.equal(h.requests[0].signal?.aborted,true,'network requests are cancelled too');
  for(const d of h.requests) d.resolve('Too late');
  await tick();
  assert.deepEqual(h.updates,[]);
  assert.deepEqual(h.turns,[]);
});

test('question-answer capture still continues through pauses until manually stopped', async () => {
  const h=captureHarness();
  const capture=await h.start({onTurn:undefined,silenceMs:700});
  h.push(300,0.2); h.push(700,0);
  for(const d of h.requests) d.resolve('Question answer');
  await tick();
  assert.equal(capture.active,true);
  assert.deepEqual(h.turns,[]);
  capture.stop();
  assert.equal(capture.active,false);
});

function liveHarness(){
  const captures=[], sent=[], statuses=[], messages=[], spoken=[];
  const input={value:'', style:{}, scrollHeight:40};
  const ctx=vm.createContext({
    console, AbortController, setTimeout, clearTimeout,
    ltCallActive:true, ltSpeaking:false, ltSending:false,
    ltSttRecorder:null, ltStartingStt:false, ltVoiceEpoch:0, ltCallEpoch:0,
    ltSessionId:'test-session', ltMediaStream:{}, ltSendAbortController:null,
    LT_MSG_PLACEHOLDER:'Start a call',
    ltCurrentUtterance:null, ltFinishPlayback:null,
    ltPersona:{gender:'female'}, twinAuthToken:'test-token',
    window:{speechSynthesis:{}},
    speechSynthesis:{
      cancel(){}, getVoices:()=>[],
      speak(u){ spoken.push(u); },
    },
    SpeechSynthesisUtterance:class { constructor(text){ this.text=text; } },
    pickTwinVoice:()=>null, getTwinVoicePitch:()=>1.25,
    document:{
      createElement:()=>({appendChild(){},remove(){}}),
      getElementById:id=>id==='lt-msg-input' ? input : {appendChild(){}, scrollHeight:0},
    },
    autoGrowField(){}, ltSetVoiceStatus:message=>statuses.push(message),
    ltShowThinkingBubble:()=>({remove(){}}),
    handleExpiredAuth(){ throw new Error('Unexpected expiry'); },
    fetch:async (url,options)=>{
      sent.push(JSON.parse(options.body));
      return {ok:true,status:200,json:async()=>({reply:'Twin reply'})};
    },
    startServerVoiceCapture:async options=>{
      const controller={active:true,cancel(){ this.active=false; }};
      captures.push({options,controller}); return controller;
    },
    ltRenderChatMessage:async (role,text)=>messages.push({role,text}),
  });
  for(const name of ['ltStopListening','ltStartListening','ltSendMessage']) load(ctx,name);
  return {ctx,captures,sent,statuses,messages,spoken,input};
}

test('automatic listening submits one spoken response and rearms after the twin reply', async () => {
  const h=liveHarness();
  await h.ctx.ltStartListening();
  assert.equal(h.captures[0].options.silenceMs,1500);
  h.captures[0].options.onUpdate('My answer');
  assert.equal(h.input.value,'My answer');
  await h.captures[0].options.onTurn('My answer');
  await tick();
  assert.deepEqual(h.sent,[{session_id:'test-session',message:'My answer'}]);
  assert.equal(h.captures.length,2);
  assert.equal(h.captures[0].controller.active,false);
});

test('listening is gated while the twin speaks or generates a reply', async () => {
  const h=liveHarness();
  h.ctx.ltSpeaking=true;
  await h.ctx.ltStartListening();
  h.ctx.ltSpeaking=false; h.ctx.ltSending=true;
  await h.ctx.ltStartListening();
  assert.equal(h.captures.length,0);
});

test('the next user turn waits for the entire twin playback, not just the HTTP response', async () => {
  const h=liveHarness(), playback=deferred();
  h.ctx.ltRenderChatMessage=async role=>{ if(role==='twin') await playback.promise; };
  await h.ctx.ltStartListening();
  const sending=h.captures[0].options.onTurn('My answer');
  await tick();
  assert.equal(h.captures.length,1);
  assert.equal(h.ctx.ltSending,true);
  playback.resolve();
  await sending;
  assert.equal(h.captures.length,2);
});

test('reply generation errors are visible and automatically resume listening', async () => {
  const h=liveHarness();
  h.ctx.fetch=async()=>({ok:false,status:502,json:async()=>({error:'Upstream unavailable'})});
  await h.ctx.ltSendMessage('My answer');
  assert.ok(h.statuses.some(message=>message.includes('Upstream unavailable')));
  assert.equal(h.captures.length,1);
  assert.deepEqual(h.messages,[{role:'user',text:'My answer'}]);
});

test('stale callbacks from a previous turn cannot submit or overwrite the next turn', async () => {
  const h=liveHarness();
  await h.ctx.ltStartListening();
  const old=h.captures[0].options;
  h.ctx.ltStopListening();
  await h.ctx.ltStartListening();
  old.onUpdate('Stale');
  await old.onTurn('Stale');
  assert.equal(h.input.value,'');
  assert.deepEqual(h.sent,[]);
});

test('an outstanding capture setup is cancelled if the call ends before it resolves', async () => {
  const h=liveHarness(), pending=deferred();
  let cancelled=false;
  h.ctx.startServerVoiceCapture=()=>pending.promise;
  const starting=h.ctx.ltStartListening();
  h.ctx.ltCallActive=false;
  h.ctx.ltStopListening();
  pending.resolve({active:true,cancel(){cancelled=true;}});
  await starting;
  assert.equal(cancelled,true);
  assert.equal(h.ctx.ltSttRecorder,null);
});

test('twin playback waits for TTS completion before callers can resume listening', async () => {
  const h=liveHarness();
  load(h.ctx,'ltRenderChatMessage');
  let done=false;
  const playback=h.ctx.ltRenderChatMessage('twin','Your question').then(()=>{done=true;});
  assert.equal(h.ctx.ltSpeaking,true);
  assert.equal(done,false);
  await h.ctx.ltStartListening();
  assert.equal(h.captures.length,0);
  h.spoken[0].onend();
  await playback;
  assert.equal(h.ctx.ltSpeaking,false);
});

test('TTS errors are visible and release the speaking gate', async () => {
  const h=liveHarness();
  load(h.ctx,'ltRenderChatMessage');
  const playback=h.ctx.ltRenderChatMessage('twin','Your question');
  h.spoken[0].onerror({error:'synthesis-failed'});
  await playback;
  assert.equal(h.ctx.ltSpeaking,false);
  assert.ok(h.statuses.some(message=>message.includes('voice')));
});

test('a reply from an ended call cannot play in a newly started call', async () => {
  const h=liveHarness(), pending=deferred();
  h.ctx.fetch=()=>pending.promise;
  const sending=h.ctx.ltSendMessage('Old answer');
  h.ctx.ltCallEpoch++;
  h.ctx.ltSending=false;
  const newController=new AbortController();
  h.ctx.ltSendAbortController=newController;
  pending.resolve({ok:true,status:200,json:async()=>({reply:'Old reply'})});
  await sending;
  assert.deepEqual(h.messages,[{role:'user',text:'Old answer'}]);
  assert.equal(h.ctx.ltSendAbortController,newController);
  assert.equal(h.captures.length,0);
});

test('ending a call settles pending playback and cancels capture and reply generation', async () => {
  const h=liveHarness();
  load(h.ctx,'ltRenderChatMessage'); load(h.ctx,'ltStopVoice');
  const playback=h.ctx.ltRenderChatMessage('twin','Still speaking');
  const controller=new AbortController();
  h.ctx.ltSendAbortController=controller;
  h.ctx.ltCallActive=false;
  h.ctx.ltStopVoice();
  await playback;
  assert.equal(h.ctx.ltSpeaking,false);
  assert.equal(controller.signal.aborted,true);
  assert.equal(h.ctx.ltSttRecorder,null);
});

function transcriptionHarness(){
  const request=deferred();
  let signal, timeout, cleared=false;
  const ctx=vm.createContext({
    AbortController, twinAuthToken:'test-token',
    arrayBufferToBase64:()=>'', handleExpiredAuth(){},
    setTimeout(fn,ms){ timeout={fn,ms}; return 1; },
    clearTimeout(){ cleared=true; },
    fetch:async (url,options)=>{ signal=options.signal; return request.promise; },
  });
  load(ctx,'transcribeWav');
  return {ctx,request,get signal(){return signal;},get timeout(){return timeout;},get cleared(){return cleared;}};
}

test('transcription network requests have a 15-second timeout', async () => {
  const h=transcriptionHarness();
  const transcribing=h.ctx.transcribeWav(new ArrayBuffer(0));
  assert.equal(h.timeout.ms,15000);
  h.timeout.fn();
  assert.equal(h.signal.aborted,true);
  h.request.reject(new Error('aborted'));
  await assert.rejects(transcribing,/aborted/);
  assert.equal(h.cleared,true);
});

test('capture cancellation propagates to the transcription HTTP request', async () => {
  const h=transcriptionHarness(), controller=new AbortController();
  const transcribing=h.ctx.transcribeWav(new ArrayBuffer(0),controller.signal);
  controller.abort();
  assert.equal(h.signal.aborted,true);
  h.request.reject(new Error('cancelled'));
  await assert.rejects(transcribing,/cancelled/);
});

test('invalid transcription payloads are surfaced rather than treated as successful silence', async () => {
  const h=transcriptionHarness();
  const transcribing=h.ctx.transcribeWav(new ArrayBuffer(0));
  h.request.resolve({ok:true,status:200,json:async()=>({text:null})});
  await assert.rejects(transcribing,/Invalid transcription response/);
});
