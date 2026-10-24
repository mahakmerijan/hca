const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const root = path.join(__dirname, '..');
const template = fs.readFileSync(path.join(root, 'templates/index.html'), 'utf8');
const catalog = fs.readFileSync(path.join(root, 'agent/live/use_cases.py'), 'utf8');
const useCases = [...catalog.matchAll(/"id": "([^"]+)",\s*"label": "([^"]+)"/g)]
  .map(([, id, label]) => ({id, label}));

function loadForm(useCase = 'Job Interviews', useCaseId = 'job_interviews'){
  const elements = new Map();
  const context = vm.createContext({
    URL,
    selectedUseCaseLabel: useCase,
    selectedUseCaseId: useCaseId,
    twinFormData: {answer: 'PRIVATE ANSWER'},
    hasExistingTwin: false,
    hasDraftLoaded: false,
    document: {
      getElementById(id){
        if(!elements.has(id)) elements.set(id, {innerHTML: '', style: {}, value: 'PRIVATE DESCRIPTION'});
        return elements.get(id);
      },
    },
  });
  // Evaluate the real standalone renderers without booting authentication or media.
  for(const name of ['escapeHtml', 'getQuestionResearchSites', 'buildQuestionSearchUrl', 'renderQuestionResearchLink', 'refreshQuestionResearchLinks', 'renderQuestionHeading', 'renderScenarioQuestions', 'renderTwinForm', 'ltRenderQuestions']){
    const declaration = template.match(new RegExp(`^function ${name}\\([^]*?^}`, 'm'));
    if(declaration) vm.runInContext(declaration[0], context);
  }
  return {context, elements};
}

function suggestion(html){
  const anchor = html.match(/<a\b[^>]*class="q-suggestions"[^>]*>Suggestions<\/a>/);
  assert.ok(anchor, 'Every question should have a Suggestions link');
  const href = anchor[0].match(/href="([^"]+)"/)[1].replace(/&amp;/g, '&');
  return {anchor: anchor[0], url: new URL(href)};
}

function sourceTerms(id, label){
  return ` (${expectedDomains[id].join(' OR ')} OR "${label}")`;
}

test('all inline application scripts remain valid JavaScript', () => {
  for(const script of template.matchAll(/<script\b[^>]*>([^]*?)<\/script>/g)){
    new vm.Script(script[1]);
  }
});

for(const {id, label} of useCases){
  test(`live questions use their own question and the ${label} context`, () => {
    const {context, elements} = loadForm(label, id);
    const questions = [
      {id: 'name', question: "What's this person's name and title?", required: true},
      {id: 'style', question: 'How formal is their communication style?'},
    ];
    context.ltRenderQuestions(questions);
    const html = elements.get('lt-q-container').innerHTML;
    const items = html.split('<div class="q-item">').slice(1);
    assert.equal(items.length, questions.length);
    items.forEach((item, index) => {
      const {url} = suggestion(item);
      assert.equal(url.origin + url.pathname, 'https://www.google.com/search');
      assert.equal(url.searchParams.get('q'), `${label} PRIVATE DESCRIPTION ${questions[index].question}${sourceTerms(id,label)}`);
      assert.equal([...item.matchAll(/<a\b/g)].length, 1);
    });
  });
}

test('Suggestions opens a separate tab without sharing a referrer or opener', () => {
  const {context, elements} = loadForm();
  context.ltRenderQuestions([{id: 'answer', question: 'What should I research?', required: true}]);
  const html = elements.get('lt-q-container').innerHTML;
  const {anchor, url} = suggestion(html);
  assert.match(anchor, /target="_blank"/);
  assert.match(anchor, /rel="noopener noreferrer"/);
  assert.match(anchor, /referrerpolicy="no-referrer"/);
  assert.match(anchor, /title="[^"]*Google[^"]*new tab[^"]*"/);
  assert.ok(url.searchParams.get('q').includes('PRIVATE DESCRIPTION'));
  assert.ok(!url.searchParams.get('q').includes('PRIVATE ANSWER'));
  assert.match(html, /oninput="twinFormData\['answer'\]=this.value"/);
  assert.match(html, /class="mic-btn"/);
  assert.match(html, /\*<\/span>/);
  assert.equal(elements.get('lt-build-row').style.display, 'flex');
});

test('question text and URL are escaped, including Unicode and query delimiters', () => {
  const {context, elements} = loadForm('Research & Development', null);
  const question = 'How about "R&D" + 50%? <img src=x onerror=alert(1)> \u65e5\u672c\u8a9e';
  context.ltRenderQuestions([{id: 'encoding', question}]);
  const html = elements.get('lt-q-container').innerHTML;
  const {url} = suggestion(html);
  assert.equal(url.searchParams.get('q'), `Research & Development PRIVATE DESCRIPTION ${question}`);
  assert.equal([...url.searchParams].length, 1);
  assert.ok(!html.includes('<img'));
  assert.ok(html.includes('&lt;img'));
});

test('scenario questions of every input type have Suggestions links', () => {
  const {context, elements} = loadForm('Venture Capital & Investor Pitching');
  context.renderScenarioQuestions([
    {id: 'scale', question: 'How analytical are they?', type: 'scale'},
    {id: 'select', label: 'What is their focus?', type: 'single_select', options: []},
    {id: 'multi', question: 'What do they value?', type: 'multi_select', options: []},
    {id: 'long', question: 'What is their background?', type: 'textarea'},
    {id: 'short', question: 'What is their title?', type: 'text'},
  ]);
  const items = elements.get('scenario-q-container').innerHTML.split('<div class="q-item">').slice(1);
  assert.equal(items.length, 5);
  items.forEach(item => assert.ok(suggestion(item).url.searchParams.get('q').startsWith('Venture Capital & Investor Pitching ')));
});

test('profile questions use the profile section, not an unrelated selected use case', () => {
  const {context, elements} = loadForm('Dating a Woman');
  context.renderTwinForm({sections: [{
    title: 'Communication preferences',
    questions: [{id: 'answer', label: 'How do you resolve conflict?', type: 'textarea'}],
  }]});
  const html = elements.get('twin-q-container').innerHTML;
  assert.equal(suggestion(html).url.searchParams.get('q'), 'Personal profile Communication preferences How do you resolve conflict?');
  assert.ok(html.includes('PRIVATE ANSWER'));
  assert.ok(!html.includes('q-source-link'));
});

const expectedDomains = {
  dating_woman: ['bumble.com', 'hinge.co', 'gottman.com'],
  job_interviews: ['ambitionbox.com', 'glassdoor.com', 'indeed.com'],
  girlfriend_conversation: ['gottman.com', 'psychologytoday.com', 'relate.org.uk'],
  parents_conversation: ['psychologytoday.com', 'greatergood.berkeley.edu', 'aamft.org'],
  regulatory_compliance: ['sec.gov', 'fca.org.uk', 'theiia.org'],
  medical_consultations: ['ama-assn.org', 'aafp.org', 'nice.org.uk'],
  vc_investor_pitching: ['ycombinator.com', 'sequoiacap.com', 'a16z.com'],
  crisis_management_pr: ['prsa.org', 'cipr.co.uk', 'instituteforpr.org'],
  diplomatic_negotiation: ['pon.harvard.edu', 'culturalatlas.sbs.com.au', 'unitar.org'],
  corporate_leadership_boardrooms: ['hbr.org', 'nacdonline.org', 'iod.com'],
};

for(const {id, label} of useCases){
  test(`${label} combines recommended domains and a broader topic in just one Suggestions link`, () => {
    const {context, elements} = loadForm(label, id);
    const question = 'What should I know about their expectations?';
    context.ltRenderQuestions([{id: 'expectations', question}]);
    const html = elements.get('lt-q-container').innerHTML;
    assert.equal([...html.matchAll(/<a\b/g)].length, 1);
    assert.ok(!html.includes('Recommended sources'));
    assert.ok(!html.includes('q-source-link'));
    const query=suggestion(html).url.searchParams.get('q');
    assert.equal(query, `${label} PRIVATE DESCRIPTION ${question}${sourceTerms(id,label)}`);
    assert.ok(!query.includes('site:'));
    for(const [otherId, domains] of Object.entries(expectedDomains)){
      if(otherId!==id) domains.filter(domain=>!expectedDomains[id].includes(domain))
        .forEach(domain=>assert.ok(!query.includes(domain)));
    }
  });
}

test('description is encoded safely and whitespace-only descriptions are omitted', () => {
  const {context} = loadForm();
  const description = 'R&D "interview" + 50% <script>alert(1)</script> \u65e5\u672c\u8a9e';
  const query = new URL(context.buildQuestionSearchUrl('What next?', 'Job Interviews', description));
  assert.equal(query.searchParams.get('q'), `Job Interviews ${description} What next?`);
  assert.equal([...query.searchParams].length, 1);
  assert.equal(new URL(context.buildQuestionSearchUrl('What next?', 'Job Interviews', '  ')).searchParams.get('q'), 'Job Interviews What next?');
});

test('unknown use cases retain broad research without unrelated website recommendations', () => {
  const {context, elements} = loadForm('New Use Case', 'new_use_case');
  context.ltRenderQuestions([{id:'new', question:'What should I research?'}]);
  const html = elements.get('lt-q-container').innerHTML;
  assert.equal(suggestion(html).url.searchParams.get('q'), 'New Use Case PRIVATE DESCRIPTION What should I research?');
  assert.ok(!html.includes('q-source-link'));
});

test('description cannot break out of research link attributes', () => {
  const {context, elements} = loadForm();
  const description = '"><img src=x onerror=alert(1)> & role';
  context.document.getElementById('lt-desc-input').value = description;
  context.ltRenderQuestions([{id:'safe', question:'What should I research?'}]);
  const html = elements.get('lt-q-container').innerHTML;
  assert.ok(!html.includes('<img'));
  assert.ok(suggestion(html).url.searchParams.get('q').includes(description));
  assert.equal([...html.matchAll(/<a\b/g)].length, 1);
});

test('switching use cases replaces both the description and recommended sources', () => {
  const {context, elements} = loadForm();
  const question = [{id:'switch', question:'What communication style should I expect?'}];
  context.ltRenderQuestions(question);
  context.selectedUseCaseId = 'dating_woman';
  context.selectedUseCaseLabel = 'Dating a Woman';
  context.document.getElementById('lt-desc-input').value = 'A first date at a cafe';
  context.ltRenderQuestions(question);
  const html = elements.get('lt-q-container').innerHTML;
  assert.ok(suggestion(html).url.searchParams.get('q').startsWith('Dating a Woman A first date at a cafe '));
  assert.ok(suggestion(html).url.searchParams.get('q').includes('bumble.com OR hinge.co OR gottman.com'));
  assert.ok(!html.includes('glassdoor.com'));
  assert.ok(!html.includes('PRIVATE'));
});

test('editing the description preserves combined source hints without changing answers', () => {
  const {context, elements} = loadForm();
  const links = [{
    dataset: {researchQuestion: 'What is the interview style?', researchContext: 'Job Interviews', researchUseCase: 'job_interviews'},
    href: '',
  }];
  elements.set('lt-q-container', {querySelectorAll: () => links});
  context.refreshQuestionResearchLinks('lt-q-container', 'New company & role');
  links.forEach(link => assert.equal(new URL(link.href).searchParams.get('q'),
    `Job Interviews New company & role What is the interview style?${sourceTerms('job_interviews','Job Interviews')}`));
  assert.equal(context.twinFormData.answer, 'PRIVATE ANSWER');
  assert.match(template, /oninput="refreshQuestionResearchLinks\('lt-q-container',this.value\)"/);
});
