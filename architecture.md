Read the whole codebase . In the current codebase 

I am getting  a video from a person – the codebase is acting like AI communication coach , I am analyzing the video and, in the output, giving the personality assessment of the person- 
 Coach's #1 Priority for You 
3-Layer Communication Analysis:- Verbal Intelligence, Non-Verbal Intelligence, Behavioral Signals
Your Strengths & Weaknesses
Your Success Probabilities of Closing a Deal, Getting Promoted , delivering a presentation, Going on a date
Your Personal Improvement Plan :- Verbal Authority & Structure, Vocal Presence
 Key Moments: Where Your Expressions Were Off – why it’s off , how to deliver it better (based on posture , facial micro expressions, eye movement, body language , voice tone) 
All these above mentioned are already happening
Now I want you to keep everything above mentioned intact. With those I want you to add the following features:-
I want you to create a form – which the user will be told to fill up regarding the behavioral  traits and other daily activities 
I want to you to create a digital twin of the person  based on the behaviour, posture , voice and everything – now it will create  nearly thousand situations of dating scenarios, job interviews, now the digital twin of the person will talk to those  thousand  other AI based digital twins of people and after talking to them it will give the result that which part can be worked out more for improvement to get a date, to get a job , to get good investment for a business idea- after that it will say which scenarios got success  and which ones were failures – regarding the failures it will tell – in which scenarios it got failures and how the person in the video can improve it.

Now to implement it I want you to go in the following direction :-

1.	Keep an optional area in the front end ‘Make a Digital twin’ – 
Make a form – which the user needs to feel – the form will have every kind of questionnaire regarding the user about the following things:-

a.	 Behavioral Model (How you act)
•	Personality traits (introvert/extrovert, risk-taking, etc.) 
•	Habits & routines 
•	Communication style 
•	Decision patterns
b.	Cognitive / Decision Model (How you think)
•	Career preferences 
•	Dating preferences 
•	Risk tolerance (business/investment) 
•	Values & goals
Based on this ‘a’ and ‘b’ section – the whole questionnaire will be created and a profile will be created – it will be an LLM based Persona ( the LLM is the gemini 2.5 pro – we are using in the .env file) can be created . It will :- 
•	Create a structured persona profile 
•	Use prompt engineering + memory 
Example:
You are a digital twin of [User].
Traits:
- Slightly introverted
- Analytical thinker
- Avoids confrontation
- Interested in startups

Communication style:
- Short responses
- Hesitant in emotional topics
👉 This becomes your base twin

c.	Embodied Model (How you present yourself)
•	Posture 
•	Facial expressions 
•	Body language
•  Spine angle 
•  Eye contact 
•  Hand movement frequency 
•  Confidence indicators
         This is something you were already gathering from the video before – keep doing it using the folders ‘external_repos/openpose’ and ‘external_repos/mediapipe’ 
2.	Use the current Gemini 2.5 pro LLM as mentioned in the .env
3.	For memory system use Vector DBs -  Pinecone and Weaviate – among them check the inference – which one is performing faster and after evaluation which one is giving more accurate result – based on that choose the proper vector DB
4.	For multi agent simulation engine :- use Langchain  and LangGraph framework
5.	What happens here:
•	Your twin interacts with: 
o	“Recruiter agent” 
o	“Date agent” 
o	“Investor agent” 
Each agent has:
•	Different personality 
•	Different goals 
 
6.	 Feedback Engine
This analyzes interactions:
Example outputs:
•	“You interrupted too often” 
•	“You showed low confidence” 
•	“Your answers lacked clarity for investors” 
 
7.	For datasets:- use all the folders and files  in the ‘datasets’ folder for training 

8.	Domain-Specific Interaction Data
Dating:
•	Conversation patterns 
•	Attraction signals 
Jobs:
•	Interview datasets 
•	HR evaluation rubrics 
Business:
•	Pitch decks 
•	Investor Q&A transcripts


How the Simulation Works (Example)
Scenario: Job Interview
1.	Your twin enters simulation 
2.	Recruiter agent asks questions 
3.	Twin responds based on your behavior 
4.	System tracks: 
o	Confidence 
o	Clarity 
o	Logical structure 
👉 Output:
•	Weak storytelling 
•	Low assertiveness 
•	Improve eye contact


Here is the architecture:-


                          ┌──────────────────────────────┐
                          │        USER INPUT LAYER      │
                          └──────────────────────────────┘
                                      │
        ┌───────────────┬─────────────┴─────────────┬───────────────┐
        │               │                           │               │
   Text / Chat     Video Capture               Audio Input     External Data
 (WhatsApp, etc.) (Camera / Upload)         (Mic recordings) (LinkedIn, etc.)
        │               │                           │               │
        └───────────────┴─────────────┬─────────────┴───────────────┘
                                      │
                          ┌──────────────────────────────┐
                          │     DATA PROCESSING LAYER    │
                          └──────────────────────────────┘
                                      │
        ┌───────────────┬─────────────┼─────────────┬───────────────┐
        │               │             │             │               │
  NLP Processing   Pose Estimation  Speech Analysis Emotion Detection  Feature Extraction
        │               │             │             │               │
        │               │             │             │               │
        ▼               ▼             ▼             ▼               ▼
   Text Embeddings   Body Metrics   Tone/Speed   Emotion Tags   Unified Feature Vector
                                      │
                                      ▼
                          ┌──────────────────────────────┐
                          │      PROFILE BUILDER         │
                          └──────────────────────────────┘
                                      │
                          Builds Structured User Profile:
                          - Personality Traits
                          - Behavior Patterns
                          - Communication Style
                          - Confidence Scores
                                      │
                                      ▼
                          ┌──────────────────────────────┐
                          │     DIGITAL TWIN ENGINE      │
                          └──────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
  Persona Model               Memory System                 Decision Model
 (LLM-based twin)        (Vector DB + history)        (Goals, preferences)
        │                             │                             │
        └───────────────┬─────────────┴─────────────┬───────────────┘
                        │                           │
                        ▼                           ▼
              ┌──────────────────┐        ┌────────────────────┐
              │ CONTEXT BUILDER  │        │ STATE TRACKER      │
              └──────────────────┘        └────────────────────┘
                        │
                        ▼
              ┌──────────────────────────────┐
              │ MULTI-AGENT SIMULATION LAYER │
              └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        │               │               │               │
   Recruiter Agent   Dating Agent   Investor Agent   Custom Agents
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                        │
                        ▼
              ┌──────────────────────────────┐
              │ INTERACTION SIMULATION LOOP  │
              └──────────────────────────────┘
                        │
              Tracks:
              - Response quality
              - Confidence
              - Persuasion
              - Emotional alignment
                        │
                        ▼
              ┌──────────────────────────────┐
              │     EVALUATION ENGINE        │
              └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        │               │               │               │
  Behavior Scoring  Communication   Posture Analysis  Decision Quality
                      Analysis
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                        │
                        ▼
              ┌──────────────────────────────┐
              │      FEEDBACK GENERATOR      │
              └──────────────────────────────┘
                        │
              Generates Insights:
              - Weakness areas
              - Improvement suggestions
              - Scenario-specific advice
                        │
                        ▼
              ┌──────────────────────────────┐
              │       USER DASHBOARD         │
              └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────────────────────┐
        │               │                               │
  Visual Analytics   Simulation Replay         Personalized Coaching



  In the second layer do not take the 'text/chat' part - implement the rest as it is.

  

Implementation Workflow

Step 1: Scenario Generation
Create a "Target Twin Generator." For each of the 10 scenarios, the AI generates a unique counter-party:

Job Interview: 3 variations (example:- The "Hardball" Manager, The "Friendly" HR, The "Bored" Executive).

Investor Pitch: 3 variations (example:- The "Numbers-Only" VC, The "Visionary" Angel).

Dating: 4 variations (example:- The "Introverted" Match, The "High-Energy" Match).

Step 2: The "Social Ping" Loop
Rather than a full 20-minute conversation for each (which is too expensive), use Social Micro-Simulations:

Opening: Twin delivers the pitch/intro.

Reaction: Target Twin responds based on their profile.

Conflict/Pivot: Target Twin throws a "curveball" question.

Closing: Twin attempts to seal the deal.

Step 3: Success/Failure Scoring
At the end of each micro-interaction, a "Referee Agent" (a separate LLM instance) grades the interaction on a scale of 1–10 based on:

Alignment: Did the Twin’s posture/tone match the Target’s preference?

Friction: Where did the conversation stall?

Outcome: Would this Target Twin "say yes"?

4. Analysis & Feedback (The Output)
After the 10 runs, you perform Failure Cluster Analysis.

The Success Rate: "Your Digital Twin secured a 'Second Date' in 42% of scenarios."

The "Why": Instead of random feedback, you group failures. "In 80% of your failed investor pitches, the failure happened exactly when they asked about 'Risk'—your Twin’s 'Verbal Authority' dropped by 30%."

The Comparison: "You are highly successful with 'Analytical' personalities but fail 90% of the time with 'Emotional' personalities." 

Also I want you to follow the caching strategy :-

What you cache:

System Instructions: The behavioral rules of the "Coach."

The Digital Twin Persona: The full JSON of the user’s behavior, posture, and voice analysis.

Success Criteria: The logic used to determine if a date or job interview was "successful."




Implementation Workflow for caching:- 
Step A: Create the Cache (The "Heavy" Lift)
When the user finishes the video analysis and form, you generate the "Digital Twin Blueprint" and send it to the cache.

Python
# Pseudo-code for Explicit Caching
cache = genai.create_context_cache(
    model='models/gemini-2.5-pro',
    system_instruction="You are a Digital Twin of [User Name]...",
    contents=[video_analysis_report, behavior_json, user_form_data],
    ttl=datetime.timedelta(minutes=15) # Keep alive only for the simulation run
)
Step B: Execute the 10 "Light" Calls
Now, you run your 10 simulations. Each call only sends the unique scenario data (e.g., "This is a job interview with a hostile boss").

Python
# Each of the 10 calls now looks like this:
response = model.generate_content(
    "Scenario 482: Pitching to a skeptical Angel Investor.",
    cached_content=cache.name
)


Use Redis Caching as well for everything - since redis is free










  Microservices Architecture:-

                        ┌──────────────────────┐
                        │   API GATEWAY        │
                        └──────────────────────┘
                                   │
     ┌──────────────┬──────────────┼──────────────┬──────────────┐
     │              │              │              │              │
┌───────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ User Svc  │ │ Twin Svc   │ │ Simulation │ │ Analysis   │ │ Media Svc  │
│           │ │            │ │ Engine     │ │ Engine     │ │ (CV/Audio) │
└───────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
     │              │              │              │              │
     └──────┬───────┴──────┬───────┴──────┬───────┴──────┬──────┘
            │              │              │              │
     ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ Knowledge  |.|            | |            | |            |
     |   Graph    │ │ Vector DB  │ │ Redis      │ │ Object     │
     │ (core data)│ │ (memory)   │ │ (cache)    │ │ Storage    │
     └────────────┘ └────────────┘ └────────────┘ └────────────┘

Also use PostGreSQL wherver needed


Use Redis (Cache) for 

Session data
Active simulations
Fast retrieval






Microservices Breakdown

1. API Gateway
Auth (JWT)
Rate limiting
Routes requests to services

2. User Service

Handles:

User profiles
Authentication
Preferences

3. Twin Service (CORE)

Handles:

Persona creation
Memory retrieval
Twin generation (LLM prompts)

4. Simulation Engine

Handles:

Multi-agent orchestration
Scenario execution

5. Analysis Engine

Handles:

Scoring
Feedback generation
Improvement insights

6. Media Service

Handles:

Video → posture
Audio → tone/emotion





API Design (Important)

Auth

POST /auth/register
POST /auth/login

Twin APIs

POST /twin/create
GET  /twin/{user_id}
PUT  /twin/update

Simulation APIs

POST /simulation/start
GET  /simulation/{id}
POST /simulation/{id}/step

Feedback APIs

GET /analysis/{simulation_id}
GET /insights/{user_id}

Media APIs

POST /media/upload-video
POST /media/upload-audio
GET  /media/{id}/analysis


First read the Deep Agents architecture from the below link :-
https://docs.langchain.com/oss/python/deepagents/overview

Next read the context engineering of Deep Agents from the below link:- 
https://docs.langchain.com/oss/python/deepagents/context-engineering

Next for the Feedback part, use the code from 'deepagents' folder - here the Langgraph deepAgents will be used using 'agent harness' procedure. take the decision accordingly after reading those links.


Also regarding the feedback we need Memory implementation in Langgraph , and for that you need to follow the below  links:-
https://docs.langchain.com/oss/python/concepts/memory

and 

https://docs.langchain.com/oss/python/langgraph/add-memory

