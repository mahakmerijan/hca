function getGuidedTourSections(screen){
  const step=(target,title,text)=>({target,title,text});
  const sections={
    welcome1:[{id:'welcome',steps:[
      step('#welcome-step1 .menu-frame','Welcome to Vyakti','Practice a real conversation with a digital twin, then get coaching. Tap the welcome image to continue after this tutorial.'),
    ]}],
    'twin-intro-video':[{id:'intro-video',steps:[
      step('#twin-intro-video','Meet your digital twin','This introduction explains the experience. The video is paused while this guide is open.'),
      step('#twin-intro-overlay .welcome-skip','Continue at your own pace','Use Skip to move past the video. Skipping this tutorial does not skip the video.'),
    ]}],
    'twin-intro-image':[{id:'intro-image',steps:[
      step('#twin-intro-image .menu-frame','Prepare for your conversation','You will choose a situation and describe the person you are preparing to face. Tap this image to continue after closing the guide.'),
    ]}],
    'use-case-select':[{id:'use-cases',waitFor:'#use-case-grid .uc-card',steps:[
      step('#use-case-grid .uc-card','Choose what to practice','Click the situation that fits your goal. Your twin, follow-up questions, and coaching will be tailored to that use case.'),
      step('#use-case-grid .uc-level-pill','Track your practice','Each situation has its own level and practice streak. These update as you complete sessions.'),
      step('#btn-logout','Log out safely','Use Log Out when you are done, especially on a shared device.'),
    ]}],
    'twin:tab-form':[
      {id:'describe',steps:[
        step('#lt-desc-input','Describe the person','Write who you are about to face and what the conversation is about. Include the role, company or relationship, and your goal.'),
        step('#lt-gen-q-btn','Generate tailored questions','After entering a description, click here for follow-up questions specific to your situation. This guide never submits your answers for you.'),
        step('#uc-banner','Change your practice situation','Use Change use case to choose another situation. Changing use case clears the current preparation and call.'),
      ]},
      {id:'questions',steps:[
        step('#lt-q-container .q-chips','Choose the twin voice','Answer the required Male or Female question. This selects the voice used by your twin during the call.'),
        step('#lt-q-container textarea','Answer the follow-ups','Write what you know about the person. Your answers help build a more relevant twin.'),
        step('#lt-q-container .q-suggestions','Research with Suggestions','Click Suggestions beside a question to open a Google search tailored to that question, your description, and the use case. Avoid including confidential details.'),
        step('#lt-q-container .mic-btn','Speak an answer','Click a question microphone to dictate instead of typing. Click it again to stop. These question buttons are separate from the hands-free live call.'),
        step('#lt-build-btn','Build their digital twin','Once you have answered the required question and provided useful details, click here to build the twin and move to Live Conversation.'),
      ]},
    ],
    'twin:tab-video':[{id:'live-call',steps:[
      step('#lt-persona-banner','Check who you will face','Review the twin name, role, and personality before starting the call.'),
      step('#lt-start-call-btn','Start your live call','Click Start Live Call and allow camera and microphone access. Keep your face visible and choose a quiet space.'),
      step('#lt-msg-input','Speak naturally, without buttons','Wait for the twin to finish and for the Listening status. Speak your answer; a 1.5-second pause submits it automatically. Transcription and reply generation add some delay.'),
      step('#lt-voice-status','Follow the conversation status','This tells you when the twin is speaking, when to speak, and when your words are being processed. Speech during the twin turn is not submitted.'),
      step('#lt-start-call-btn','Finish and get coaching','During the call, an End & Get Judged button appears beside Start. Click it when you are finished to stop the call and generate your report. Tutorials stay out of active calls.'),
    ]}],
    'twin:tab-results':[
      {id:'results-empty',waitFor:'#lt-judgment-empty',steps:[
        step('#lt-judgment-empty','Your coaching will appear here','Complete a Live Conversation and click End & Get Judged to see your personalized report.'),
        step('[data-tour="live-tab"]','Return to your conversation','Use Live Conversation to start a call after building a twin. You can move between the tabs before starting.'),
      ]},
      {id:'judgment',steps:[
        step('[data-tour="judgment-summary"]','Review your session','Start with the summary to understand how you handled this situation.'),
        step('[data-tour="confidence-score"]','Understand your confidence score','Review the composite score out of 100, then read the explanations below rather than judging yourself by a number alone.'),
        step('[data-tour="growth-hexagon"]','Explore your growth areas','The hexagon shows six conversation skills. Use it to choose an area to practice next.'),
        step('[data-tour="practice-streak"]','See your practice progress','Your streak and level belong to this use case. Consistent practice helps you work toward a tougher twin.'),
        step('[data-tour="score-breakdown"]','Find specific improvements','Read the score explanations and any moment-by-moment feedback, strengths, and coaching tips below.'),
        step('#uc-banner','Practice again','Choose Describe Them to prepare another twin, or Change use case for a different situation.'),
      ]},
    ],
  };
  return typeof screen==='string' && Object.prototype.hasOwnProperty.call(sections,screen)
    ? sections[screen] : [];
}

function createGuidedTour({document,window,storage,isAuthenticated,isCallActive,onPause=()=>{},onResume=()=>{}}){
  const el=id=>document.getElementById(id);
  const overlay=el('tour-overlay'), dialog=el('tour-dialog'), launch=el('tour-launch');
  const remembered=new Set();
  let screen=null, tour=null, index=0, frame=null, previousFocus=null, storageFailed=false;

  function storageWarning(error){
    if(!storageFailed) console.warn('[Tutorial] Browser storage unavailable:',error);
    storageFailed=true;
    el('tour-storage-note').textContent='This browser cannot remember tutorials between visits. You can still use or skip this guide.';
  }
  function hasSeen(id){
    if(remembered.has(id)) return true;
    try{ return storage.getItem('vyakti.tour.v1.'+id)==='done'; }
    catch(error){ storageWarning(error); return false; }
  }
  function remember(id){
    remembered.add(id);
    try{ storage.setItem('vyakti.tour.v1.'+id,'done'); }
    catch(error){ storageWarning(error); }
  }
  function target(step){
    const node=document.querySelector(step.target);
    return node && node.getClientRects().length ? node : null;
  }
  function available(){
    return getGuidedTourSections(screen)
      .filter(section=>!section.waitFor || target({target:section.waitFor}))
      .map(section=>({
      id:section.id,steps:section.steps.filter(step=>target(step)),
    })).filter(section=>section.steps.length);
  }
  function allowed(){
    return isAuthenticated() && !isCallActive() && getGuidedTourSections(screen).length>0;
  }
  function position(){
    frame=null;
    if(!tour) return;
    const anchor=target(tour.steps[index]);
    if(!anchor){ close(); return; }
    const rect=anchor.getBoundingClientRect();
    const width=window.innerWidth,height=window.innerHeight,gap=12;
    const left=Math.max(4,Math.min(rect.left-4,width-8));
    const top=Math.max(4,Math.min(rect.top-4,height-8));
    Object.assign(el('tour-spotlight').style,{
      left:left+'px',top:top+'px',
      width:Math.max(0,Math.min(rect.right+4,width-4)-left)+'px',
      height:Math.max(0,Math.min(rect.bottom+4,height-4)-top)+'px',
    });
    const box=dialog.getBoundingClientRect();
    const cardWidth=Math.min(box.width,width-2*gap);
    const below=rect.bottom+gap,above=rect.top-box.height-gap;
    const cardTop=below+box.height<=height-gap ? below
      : above>=gap ? above : Math.max(gap,height-box.height-gap);
    Object.assign(dialog.style,{
      left:Math.max(gap,Math.min(rect.left,width-cardWidth-gap))+'px',
      top:Math.max(gap,cardTop)+'px',
    });
  }
  function schedulePosition(){
    if(!tour || frame!==null) return;
    frame=window.requestAnimationFrame(position);
  }
  function render(){
    const step=tour.steps[index],anchor=target(step);
    if(!anchor){ close(); return; }
    el('tour-title').textContent=step.title;
    el('tour-description').textContent=step.text;
    el('tour-progress').textContent='Step '+(index+1)+' of '+tour.steps.length;
    el('tour-back').disabled=index===0;
    el('tour-next').textContent=index===tour.steps.length-1?'Done':'Next';
    anchor.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'});
    schedulePosition();
    el('tour-next').focus();
  }
  function close({remember:save=false,restoreFocus=true}={}){
    if(!tour) return;
    if(save) tour.ids.forEach(remember);
    tour=null;
    overlay.hidden=true;
    if(frame!==null){ window.cancelAnimationFrame(frame); frame=null; }
    onResume(screen);
    if(restoreFocus){
      const focus=previousFocus && previousFocus!==document.body && !previousFocus.disabled
        && previousFocus.isConnected && previousFocus.getClientRects().length
        ? previousFocus : launch;
      focus.focus();
    }
  }
  function open(sections){
    if(!sections.length) return;
    previousFocus=document.activeElement;
    tour={ids:sections.map(section=>section.id),steps:sections.flatMap(section=>section.steps)};
    index=0;
    overlay.hidden=false;
    onPause(screen);
    render();
  }
  function refresh(){
    launch.hidden=!isAuthenticated() || !getGuidedTourSections(screen).length;
    launch.disabled=isCallActive();
    launch.title=isCallActive()?'Tutorials resume after the call ends':'Replay the tutorial for this screen';
    if(!allowed()){ close({restoreFocus:false}); return; }
    if(tour){ if(!target(tour.steps[index])) close(); return; }
    const section=available().find(section=>!hasSeen(section.id));
    if(section) open([section]);
  }
  function start(){
    if(!allowed()) return;
    close({restoreFocus:false});
    open(available());
  }
  function next(){
    if(!tour) return;
    if(index<tour.steps.length-1){ index++; render(); }
    else { close({remember:true}); refresh(); }
  }
  function back(){
    if(tour && index>0){ index--; render(); }
  }
  function showScreen(value){
    close({restoreFocus:false});
    screen=value;
    refresh();
  }
  launch.addEventListener('click',start);
  el('tour-next').addEventListener('click',next);
  el('tour-back').addEventListener('click',back);
  el('tour-skip').addEventListener('click',()=>close({remember:true}));
  document.addEventListener('keydown',event=>{
    if(!tour) return;
    if(event.key==='Escape'){ event.preventDefault();close({remember:true}); }
    if(event.key==='Tab'){
      const buttons=[el('tour-skip'),el('tour-back'),el('tour-next')].filter(button=>!button.disabled);
      const current=buttons.indexOf(document.activeElement);
      const direction=event.shiftKey?-1:1;
      const nextIndex=(current+direction+buttons.length)%buttons.length;
      event.preventDefault();
      buttons[nextIndex].focus();
    }
  });
  document.addEventListener('focusin',event=>{
    if(tour && !dialog.contains(event.target)) el('tour-next').focus();
  });
  document.addEventListener('load',schedulePosition,true);
  window.addEventListener('resize',schedulePosition);
  window.addEventListener('scroll',schedulePosition,true);
  return {showScreen,refresh,start,next,back,close,get active(){return !!tour;}};
}

if(typeof module!=='undefined' && module.exports){
  module.exports={getGuidedTourSections,createGuidedTour};
}
