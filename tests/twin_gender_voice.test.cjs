const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '..', 'templates/index.html'), 'utf8');

function load(extra = {}){
  const elements = new Map();
  const context = vm.createContext({
    URL,
    selectedUseCaseLabel: 'Job Interviews',
    selectedUseCaseId: 'job_interviews',
    twinFormData: {},
    document: {
      getElementById(id){
        if(!elements.has(id)) elements.set(id, {innerHTML: '', style: {}, value: 'desc'});
        return elements.get(id);
      },
    },
    ...extra,
  });
  for(const name of ['escapeHtml', 'getQuestionResearchSites', 'buildQuestionSearchUrl', 'renderQuestionResearchLink', 'renderQuestionHeading', 'ltRenderQuestions', 'ltMissingRequiredGender', 'pickTwinVoice', 'getTwinVoicePitch']){
    const declaration = template.match(new RegExp(`^function ${name}\\([^]*?^}`, 'm'));
    assert.ok(declaration, `${name} should exist`);
    vm.runInContext(declaration[0], context);
  }
  vm.runInContext(template.match(/^const TWIN_VOICE_HINTS = \{[^]*?^\};/m)[0] + '\nthis.TWIN_VOICE_HINTS = TWIN_VOICE_HINTS;', context);
  return {context, elements};
}

const voices = [
  {name: 'Alex', lang: 'en-US'},
  {name: 'Samantha', lang: 'en-US'},
  {name: 'Google UK English Male', lang: 'en-GB'},
  {name: 'Google UK English Female', lang: 'en-GB'},
  {name: 'Thomas', lang: 'fr-FR'},
];

test('female twin picks a female English voice', () => {
  const {context} = load();
  assert.match(context.pickTwinVoice(voices, 'female').name, /Samantha|Female/);
});

test('male twin picks a male English voice', () => {
  const {context} = load();
  assert.match(context.pickTwinVoice(voices, 'male').name, /Alex|Male/);
});

test('gender matching is case-insensitive and tolerates missing voices', () => {
  const {context} = load();
  assert.match(context.pickTwinVoice(voices, 'Female').name, /Samantha|Female/);
  assert.equal(context.pickTwinVoice([], 'male'), null);
  assert.equal(context.pickTwinVoice(voices, ''), null);
});

test('pitch fallback differs by gender only when no matching voice exists', () => {
  const {context} = load();
  assert.ok(context.getTwinVoicePitch('female', false) > 1);
  assert.ok(context.getTwinVoicePitch('male', false) < 1);
  assert.equal(context.getTwinVoicePitch('male', true), 1);
  assert.equal(context.getTwinVoicePitch('', false), 1);
});

test('choice questions render Male/Female chips that store the answer', () => {
  const {context, elements} = load();
  context.ltRenderQuestions([
    {id: 'gender', question: 'Is this person male or female?', type: 'choice', options: ['Male', 'Female'], required: true},
    {id: 'q1', question: 'Who are they?', required: true},
  ]);
  const html = elements.get('lt-q-container').innerHTML;
  assert.match(html, /<div class="q-chips"[^>]*>/);
  assert.match(html, /toggleChip\('gender','Male',this,false\)[^>]*>Male</);
  assert.match(html, /toggleChip\('gender','Female',this,false\)[^>]*>Female</);
  assert.doesNotMatch(html, /id="q-gender"[^>]*textarea|<textarea id="q-gender"/);
  assert.match(html, /<textarea id="q-q1"/);
  assert.equal((html.match(/class="q-suggestions"/g) || []).length, 1, 'only the free-text question gets Suggestions');
});

test('building is blocked until gender is answered', () => {
  const {context} = load();
  context.twinFormData = {};
  assert.equal(context.ltMissingRequiredGender(), true);
  context.twinFormData = {gender: 'Female'};
  assert.equal(context.ltMissingRequiredGender(), false);
});
