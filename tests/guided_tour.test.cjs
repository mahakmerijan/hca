const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');

const sourcePath=path.join(__dirname,'..','static','guided-tour.js');
function library(){
  assert.ok(fs.existsSync(sourcePath),'guided tour controller should exist');
  return require(sourcePath);
}

function harness({screen='twin:tab-form',storage=new Map(),targets=['#lt-desc-input','#lt-gen-q-btn'],authenticated=true}={}){
  const events={}, nodes=new Map(), frames=[];
  let activeElement=null, callActive=false, paused=0, resumed=0;
  const classes=()=>({add(){},remove(){}});
  function node(id){
    if(!nodes.has(id)) nodes.set(id,{
      id,hidden:false,disabled:false,textContent:'',style:{},classList:classes(),isConnected:true,
      getBoundingClientRect:()=>({top:100,left:20,bottom:150,right:220,width:200,height:50}),
      getClientRects:()=>[{}],
      focus(){activeElement=this;},scrollIntoView(){},
      addEventListener(type,callback){events[id+':'+type]=callback;},
      contains(el){return el && el.id.startsWith('tour-');},
    });
    return nodes.get(id);
  }
  const visible=new Set(targets);
  const document={
    getElementById:node,
    querySelector:selector=>visible.has(selector)?node(selector):null,
    addEventListener(type,callback){events['document:'+type]=callback;},
    get activeElement(){return activeElement;},
  };
  const window={
    innerWidth:1024,innerHeight:768,
    requestAnimationFrame(callback){frames.push(callback);return frames.length;},
    cancelAnimationFrame(){},
    addEventListener(type,callback){events['window:'+type]=callback;},
  };
  node('tour-dialog').getBoundingClientRect=()=>({width:340,height:240});
  const api=library().createGuidedTour({
    document,window,
    storage:storage instanceof Map?{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)}:storage,
    isAuthenticated:()=>authenticated,isCallActive:()=>callActive,
    onPause:()=>paused++,onResume:()=>resumed++,
  });
  function flush(){ while(frames.length) frames.shift()(); }
  api.showScreen(screen); flush();
  return {api,node,events,visible,storage,window,flush,setCall(value){callActive=value;api.refresh();flush();},
    get paused(){return paused;},get resumed(){return resumed;},get focused(){return activeElement;}};
}

test('first screen visit opens a highlighted tutorial with Next and Back',()=>{
  const h=harness();
  assert.equal(h.api.active,true);
  assert.equal(h.node('tour-title').textContent,'Describe the person');
  assert.equal(h.node('tour-back').disabled,true);
  h.api.next();h.flush();
  assert.equal(h.node('tour-title').textContent,'Generate tailored questions');
  h.api.back();h.flush();
  assert.equal(h.node('tour-title').textContent,'Describe the person');
});

test('finishing a screen tutorial persists completion across page loads',()=>{
  const storage=new Map(),h=harness({storage});
  h.api.next();h.api.next();h.flush();
  assert.equal(h.api.active,false);
  assert.equal(harness({storage}).api.active,false);
});

test('Skip dismisses this section, while Show tutorial can always replay it',()=>{
  const h=harness();
  h.api.close({remember:true});
  h.api.refresh();
  assert.equal(h.api.active,false);
  h.api.start();
  assert.equal(h.api.active,true);
});

test('navigation closes an unfinished tour without marking it complete',()=>{
  const h=harness();
  h.api.showScreen('login');
  assert.equal(h.api.active,false);
  assert.equal(h.node('tour-launch').hidden,true);
  h.api.showScreen('twin:tab-form');
  assert.equal(h.api.active,true);
});

test('follow-up questions get a separate first-visit tour when they actually appear',()=>{
  const h=harness();
  h.api.close({remember:true});
  h.visible.add('#lt-q-container .q-chips');
  h.visible.add('#lt-q-container .q-suggestions');
  h.visible.add('#lt-q-container .mic-btn');
  h.visible.add('#lt-build-btn');
  h.api.refresh();h.flush();
  assert.equal(h.api.active,true);
  assert.match(h.node('tour-title').textContent,/voice/i);
});

test('absent targets are skipped without marking an unavailable report as seen',()=>{
  const h=harness({screen:'twin:tab-results',targets:[]});
  assert.equal(h.api.active,false);
  assert.equal(h.storage.size,0);
  h.visible.add('[data-tour="confidence-score"]');
  h.api.refresh();h.flush();
  assert.equal(h.api.active,true);
  assert.match(h.node('tour-title').textContent,/confidence/i);
});

test('live calls suppress automatic and manual tutorials without completing them',()=>{
  const h=harness({screen:'twin:tab-video',targets:['#lt-start-call-btn','#lt-msg-input']});
  h.setCall(true);
  assert.equal(h.api.active,false);
  assert.equal(h.node('tour-launch').disabled,true);
  h.api.start();
  assert.equal(h.api.active,false);
  assert.equal(h.storage.size,0);
  h.setCall(false);
  assert.equal(h.api.active,true);
});

test('Escape skips the tutorial and focus returns to the launch button',()=>{
  const h=harness();
  let prevented=false;
  h.events['document:keydown']({key:'Escape',preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.equal(h.api.active,false);
  assert.equal(h.focused,h.node('tour-launch'));
});

test('Tab and Shift+Tab keep keyboard focus inside the tutorial',()=>{
  const h=harness();
  h.node('tour-next').focus();
  h.events['document:keydown']({key:'Tab',shiftKey:false,preventDefault(){}});
  assert.equal(h.focused,h.node('tour-skip'));
  h.events['document:keydown']({key:'Tab',shiftKey:true,preventDefault(){}});
  assert.equal(h.focused,h.node('tour-next'));
});

test('tutorial stays inside a narrow viewport on resize',()=>{
  const h=harness();
  h.window.innerWidth=320;h.window.innerHeight=480;
  h.events['window:resize']();h.flush();
  const style=h.node('tour-dialog').style;
  assert.ok(parseFloat(style.left)>=12);
  assert.ok(parseFloat(style.top)>=12);
  assert.ok(parseFloat(style.top)<=480-12);
});

test('blocked browser storage produces a visible notice and remembers the visit for this page',()=>{
  const h=harness({storage:{
    getItem(){throw new Error('Storage blocked');},setItem(){throw new Error('Storage blocked');},
  }});
  assert.match(h.node('tour-storage-note').textContent,/cannot.*remember/i);
  h.api.close({remember:true});h.api.refresh();
  assert.equal(h.api.active,false);
});

test('all reachable post-login screens have tutorials',()=>{
  for(const screen of ['welcome1','twin-intro-video','twin-intro-image','use-case-select',
    'twin:tab-form','twin:tab-video','twin:tab-results']){
    assert.ok(library().getGuidedTourSections(screen).length>0,screen);
  }
  assert.equal(library().getGuidedTourSections('login').length,0);
  assert.equal(library().getGuidedTourSections('<script>東京</script>').length,0);
});

test('each tutorial step has a static target and nonempty helpful text',()=>{
  const {getGuidedTourSections}=library();
  for(const screen of ['welcome1','twin-intro-video','twin-intro-image','use-case-select',
    'twin:tab-form','twin:tab-video','twin:tab-results']){
    for(const section of getGuidedTourSections(screen)){
      for(const step of section.steps){
        assert.equal(typeof step.target,'string');
        assert.ok(step.title.trim() && step.text.trim());
      }
    }
  }
});

test('unknown or malformed screen names cannot resolve inherited object properties',()=>{
  const {getGuidedTourSections}=library();
  for(const screen of ['constructor','__proto__','toString',null,undefined,42]){
    assert.deepEqual(getGuidedTourSections(screen),[]);
  }
});

test('no tutorial is shown or persisted before authentication',()=>{
  const h=harness({authenticated:false});
  assert.equal(h.api.active,false);
  assert.equal(h.node('tour-launch').hidden,true);
  assert.equal(h.storage.size,0);
});

test('the empty judgment tutorial does not suppress the actual report tutorial',()=>{
  const h=harness({screen:'twin:tab-results',targets:['#lt-judgment-empty']});
  h.api.close({remember:true});
  h.visible.delete('#lt-judgment-empty');
  h.visible.add('[data-tour="confidence-score"]');
  h.api.refresh();
  assert.equal(h.api.active,true);
  assert.match(h.node('tour-title').textContent,/confidence/i);
});

test('an active tutorial does not restart when asynchronous content refreshes the screen',()=>{
  const h=harness();
  h.api.next();
  h.api.refresh();
  assert.equal(h.node('tour-title').textContent,'Generate tailored questions');
});

test('use-case guidance waits for the cards instead of starting with only Log Out',()=>{
  const h=harness({screen:'use-case-select',targets:['#btn-logout']});
  assert.equal(h.api.active,false);
  h.visible.add('#use-case-grid .uc-card');
  h.api.refresh();
  assert.equal(h.node('tour-title').textContent,'Choose what to practice');
});

test('an available report never shows the empty-report instructions first',()=>{
  const h=harness({screen:'twin:tab-results',targets:[
    '[data-tour="live-tab"]','[data-tour="confidence-score"]',
  ]});
  assert.match(h.node('tour-title').textContent,/confidence/i);
});

test('closing a tutorial resumes the paused introduction',()=>{
  const h=harness({screen:'welcome1',targets:['#welcome-step1 .menu-frame']});
  assert.equal(h.paused,1);
  h.api.close({remember:true});
  assert.equal(h.resumed,1);
});

test('tour integrates dynamic content, navigation, and call start without relying on polling',()=>{
  const template=fs.readFileSync(path.join(__dirname,'..','templates/index.html'),'utf8');
  assert.match(template,/guided-tour\.js/);
  for(const name of ['appShowScreen','ltRenderQuestions','renderUseCaseGrid','ltRenderJudgment','ltStartCall']){
    const fn=template.match(new RegExp(`^(?:async )?function ${name}\\([^]*?^}`,'m'))[0];
    assert.match(fn,/guidedTour/,name);
  }
  assert.match(template,/id="tour-dialog"[^>]*role="dialog"[^>]*aria-modal="true"/);
});
