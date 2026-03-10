# Workshop Assignment: AI-Powered Training Planner for Ultimate Frisbee

## Context

Flik Ultimate (flikulti.com) is a platform with hundreds of training materials for Ultimate Frisbee coaches and players — covering tactics, drills, practice sessions, video tutorials, and strength & conditioning.

**Your team:**
- **Maartje** is the trainer and your point of contact. She can give you access to the Flik platform as a team member, so you can explore all the available content. Follow this link to be added to the team and get access: https://www.flikulti.com/team?code=092dfa1e14db869049a251e7

- **The prep work is done:** Maartje has already scraped, cleaned, and loaded all Flik training materials into a vector store. You don't need to build that part. See `TECHNICAL_DETAILS.md` for how to access and query it.

---

## The Problem

Flik has an enormous library of content — but as a trainer, finding the right material for a specific training session takes too much time and effort. You need to browse categories manually, cross-reference drills with theory, and figure out a logical progression yourself. This gets in the way of actually coaching.

---

## What We Want

A tool where a trainer can describe what they want to train — in their own words — and get back a concrete, ready-to-use training plan. Fast, with minimal effort.

The trainer should be able to:

1. **Describe their intent** — for example:
   - *"I want to work on breaking the mark with my intermediate club team"*
   - *"My players struggle with timing their cuts in a vertical stack"*
   - *"We're playing a team that runs cup zone next weekend, how do we prepare?"*

2. **Receive a concrete training plan** — not just a list of links, but a structured plan they can walk onto the pitch with, including:
   - A clear theme or objective for the session
   - Relevant drills or exercises, ordered from simple to complex
   - Coaching notes: what to look for, what to emphasise, how to know the drill is going well
   - Links back to the source material on flikulti.com

3. **Get a short-term progression** — not just one session, but a suggestion for the next 2–3 sessions in case the topic requires multiple steps, repetition, or a build-up.

---

## Understanding the Flik Content

Before building, spend some time on the site. Here's what to know about how the content is structured — it will inform how you use it.

### Theory articles
Theory articles (https://www.flikulti.com/theory/) hold the bulk of conceptual information. They link to related content in two ways:
- **Internal page links** embedded in the article body, shown as `[link id="123"]` in the raw content — these point to related theory pages or drills
- **Linked drills** listed at the foot of each article — the most directly relevant drills for that concept

For example, see: https://www.flikulti.com/theory/essentials/throwing/breaking-the-mark/ and https://www.flikulti.com/theory/analysis/rhinos-offence-line-usau-nationals-2024/

These links are your map from a concept to its practical application.

### Drill progression: prerequisites & what's next
Every drill on Flik sits within a progression. Drills have "pre-requisites" (simpler drills a player should be comfortable with first) and "what's next?" (more complex variations to build towards).

When introducing a new concept, always start with the simplest version and build up. For example, for breaking the mark:
- **Simplest:** https://www.flikulti.com/drills/cut-underneath-break/
- **Intermediate:** https://www.flikulti.com/drills/away-to-break-under/
- **Most complex:** https://www.flikulti.com/drills/team-triangle-cutting/

A good training plan respects this progression rather than jumping straight to complex patterns.

### Lesson plans as the gold standard
Flik has a set of lesson plans written by and for teachers: https://www.flikulti.com/theory/lesson-plans/

These are the most explicit examples of what a good training plan looks like — they include clear session objectives, coaching guidance, and structured progressions. Use them as your reference point for the kind of output you're aiming to produce.

### Analysis articles and real game footage
The analysis section (e.g. https://www.flikulti.com/theory/analysis/rhinos-offence-line-usau-nationals-2024/) connects theory to real game situations. If you can surface relevant analysis alongside a drill, it helps coaches understand *why* a skill matters and *how* it shows up in actual play.

---

## The AI Challenge

You are building an AI-powered assistant on top of the vector store. There are three parts to this challenge:

### 1. RAG: Find the most relevant material
Use the vector store (see `TECHNICAL_DETAILS.md`) to retrieve the Flik content that is most relevant to what the trainer describes. Think about:
- What makes a good search query to send to the vector store?
- How many results do you retrieve, and how do you filter or rank them?
- Theory articles link to drills — can you use those connections to find better drill recommendations, rather than searching for drills directly?
- Can you retrieve analysis articles that show the relevant skill in a real game context?
- How do you handle cases where the results aren't great?

### 2. Interface: Make it easy to get good input
The quality of the output depends heavily on what the trainer tells the system. Design an interface that naturally collects the right information without feeling like filling in a form. Think about:
- What does the trainer actually need to tell the system? (e.g. skill level, team size, session length, focus area)
- How do you get this from them with minimal friction?
- Could a short conversation (a few follow-up questions) improve the result?

### 3. LLM logic: Turn documents into a training plan
Once you have relevant content from the vector store, you need to use an LLM to turn it into a coherent, useful output. Think about:
- How do you structure the prompt so the LLM produces a practical training plan, not just a summary?
- How do you ensure drills are ordered from simple to complex, respecting the prerequisite structure?
- How do you get the LLM to generate coaching notes — things to emphasise, how to know the drill is going well — even though this content doesn't exist explicitly in Flik yet?
- How do you make sure the output stays grounded in the actual Flik material (with links), rather than making things up?
- Can you include a relevant example from a game analysis article to show the skill in context?

---

## What a Good Result Looks Like

A trainer types something like:

> *"I want to work on breaking the mark with my club team. They understand the basics but struggle to execute under pressure. We have 90 minutes."*

And they get back something like:

> **Session 1 – Breaking the Mark: Building the Habit**
> *Goal: Develop confident break throws from a simple pivot, starting without defensive pressure*
>
> **Warm-up (15 min):** Throwing pairs focusing on pivot range and low releases
>
> **Drill 1 (20 min):** Cut Underneath Break — [flikulti.com/drills/cut-underneath-break](https://www.flikulti.com/drills/cut-underneath-break)
> *Why this drill:* Isolates the break throw in its simplest form — no continuation required
> *Things to emphasise:* Pivot foot staying planted; leading the cutter into space, not throwing to where they are
> *How to know it's going well:* Receivers are catching without breaking stride
>
> **Drill 2 (25 min):** Away to Break Under — [flikulti.com/drills/away-to-break-under](https://www.flikulti.com/drills/away-to-break-under)
> *Progression from Drill 1:* Adds a second throw and a decision point
>
> **Themed game (20 min):** ...
> *In game footage:* See how Rhino's handler line uses break throws to reset momentum — [flikulti.com/theory/analysis/rhinos-offence-line](https://www.flikulti.com/theory/analysis/rhinos-offence-line-usau-nationals-2024/)
>
> **Further reading:** Breaking the Mark — [flikulti.com/theory/essentials/throwing/breaking-the-mark](https://www.flikulti.com/theory/essentials/throwing/breaking-the-mark/)
>
> **Session 2 – Adding Pressure**
> ...
>
> **Session 3 – Live Application**
> ...

---

## Storing the Output

Ideally, the generated practice plan should be saved directly into the trainer's Flik account at https://www.flikulti.com/sessions/, alongside their existing plans. Flik has a simple UI for building session plans by adding drills to a list — think about whether your solution can write into that format or make it easy for the coach to transfer the output there with minimal effort. If your output can also populate the **Notes** field for each drill (e.g. with the coaching guidance and links to further reading), even better.

---

## Deliverables

By the end of the workshop, you should have a working prototype that:

- Takes a trainer's description as input
- Retrieves relevant content from the Flik vector store
- Produces a structured training plan using an LLM, with drills ordered from simple to complex
- Includes coaching notes and links to further reading for each drill
- Presents the output in a way a trainer can actually use on the pitch

The interface can be as simple as a command-line prompt or as polished as a small web UI — what matters is that the end-to-end flow works and the output is genuinely useful.

---

## Bonus: Help Flik Support the Community

Flik's mission is to support the growth of Ultimate Frisbee worldwide. If this tool works, it shouldn't stay in a prototype — it should live on their website and be available to every coach who visits.

As a bonus challenge, think about how to hand this off in a way that the Flik team can actually use and maintain, without needing a team of developers.

### What this means in practice

- **Make it embeddable** — can your solution be dropped into an existing website as a simple widget or page? A lightweight web app (e.g. a single HTML page, or a small backend with a clean API) is much easier to adopt than a complex standalone system.
- **Keep dependencies minimal** — the Flik team are coaches and content creators, not engineers. The fewer moving parts, the better. If your solution requires managing a server, a database, and three API keys to keep running, it's unlikely to survive long-term.
- **Document what's needed to run it** — what API keys are required? What does it cost to run per month? What happens when the Flik content is updated — how does the vector store get refreshed?
- **Think about access** — the full Flik content library is behind a subscription. The tool on the website should work for logged-in members. How does your solution handle that?

### Why this matters

There are thousands of Ultimate Frisbee coaches around the world — club coaches, national team coaches, volunteers running youth programmes — who would benefit from a tool like this. A well-built prototype that can realistically be deployed is a meaningful contribution to a sport that runs almost entirely on volunteer effort.

You don't have to fully solve this in the workshop. But if your design choices make integration easier — clean API, no hardcoded secrets, a UI that could sit inside a WordPress page — that's a win worth aiming for.

---

## Getting Started

1. Read `TECHNICAL_DETAILS.md` — this explains how to query the vector store
2. Get access to flikulti.com via the team link above
3. Browse a few pages on the site — especially a theory article, a drill, and a lesson plan — to understand the content and how it connects
4. Start with the RAG layer: can you retrieve relevant content for a sample trainer input?
5. Then add the LLM layer: can you turn that content into a training plan?
