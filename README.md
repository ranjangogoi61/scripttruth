# ScriptTruth

**A claim verification engine for screenplays — extracts factual claims from a scene and verifies each one against live web evidence, with citations you can open yourself.**

Live app: **https://scripttruth.onrender.com**

Built for the Google Cloud **Agentic Cinema: The Blockbuster Hackathon** — Parallel track.

---

## The problem

A factual error in a screenplay is nearly free to fix at script stage and extremely expensive to fix later. The same mistake costs:

| Stage caught | Cost |
|---|---|
| Script | rewrite a line |
| Prep | re-source a prop |
| Shoot day | lost hours on set |
| Edit | cut around it, or compromise the scene |
| VFX | digital paint-out, per shot |
| After release | unrecoverable |

Yet **no role in a production formally owns external factual accuracy.** A script supervisor owns *internal* continuity — whether the coffee cup matches between takes. Whether a 1987 scene can contain a smartphone, or whether a restaurant named in dialogue existed in that decade, is checked only if the production can afford a dedicated researcher. Most can't.

ScriptTruth does that pass automatically.

---

## What makes this different from existing tools

Existing screenplay continuity tools solve the **closed-world** problem: does the script contradict *itself*? A sweater is blue in scene 12 and red in scene 18. That's answerable by reading the document carefully — no external data required.

ScriptTruth solves the **open-world** problem: does the script contradict *reality*?

```
CLOSED-WORLD (already served)        OPEN-WORLD (this project)
"sweater blue in s12, red in s18"    "1987 scene shows a smartphone"
"prop appears without setup"          "restaurant named opened in 2003"
"character in two places at once"     "law referenced didn't exist yet"
        ↓                                      ↓
   Read the document                  MUST retrieve external sources
   No search needed                   Impossible without retrieval
```

This distinction is why the project is built on a search API rather than a larger prompt. A language model cannot verify — it generates plausible text, and plausibility is not truth. Verification requires retrieval against real sources. **Remove Parallel Search from this project and there is no product left.**

---

## How it works

```
Screenplay scene
      │
      ▼
[1] CLAIM EXTRACTION  ── Gemini (google-genai)
      │  Identifies verifiable factual claims; ignores dialogue,
      │  emotion, creative description, and invented entities.
      ▼
[2] RETRIEVAL  ── Parallel Search API  (runs concurrently per claim)
      │  Live web search per claim, returning ranked sources + excerpts.
      ▼
[3] EVIDENCE COMPARISON  ── Gemini (single batched call)
      │  Compares each claim against only the evidence retrieved for it.
      ▼
[4] SCORING + CITATION  ── deterministic Python, not the model
      │  Weighted confidence score; every citation validated against
      │  the URLs actually retrieved.
      ▼
Verdict list with openable sources
```

### Verdict labels

| Label | Meaning |
|---|---|
| **Verified** | Evidence supports the claim, high confidence |
| **Likely Consistent** | Evidence supports it, moderate confidence |
| **Contradicted** | Evidence contradicts the claim, high confidence |
| **Likely Contradicted** | Evidence contradicts it, moderate confidence |
| **Disputed** | Sources genuinely disagree with each other |
| **Needs Review** | Evidence is weak or ambiguous |
| **Insufficient Evidence** | No usable public record found |

**"Insufficient Evidence" is never "false."** A fictional brand ("Stark Industries") or an invented city returns no search results — the system treats absence of evidence as absence of evidence, never as proof of error.

### Confidence scoring

Computed in Python, not asked of the model:

```
confidence = 35% source agreement      (independent corroboration)
           + 25% source authority      (how reliable the sources are)
           + 20% evidence relevance    (does it address THIS claim)
           + 20% extraction confidence (how explicit the claim was)
```

Source agreement carries the largest weight because single-source verification is the most common way fact-checking fails.

### Setting modes

The user selects the scene's world before verification: **Modern**, **Historical**, **Fictional Universe**, or **Alternate Timeline**. A fantasy scene shouldn't have its dragon flagged as factually wrong. This is a deliberate user choice rather than automatic genre detection, because a wrong guess about genre produces exactly the confident-but-wrong output the system is designed to avoid.

---

## Technologies used

| Component | Technology |
|---|---|
| Claim extraction & comparison | **Gemini** via `google-genai` (Google AI Studio) |
| Web retrieval | **Parallel Search API** (`parallel-web`) |
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | Render |

Structured outputs (`response_schema`) are used for both Gemini calls so the pipeline receives typed, predictable JSON rather than parsed prose.

---

## Running locally

**Requirements:** Python 3.10+

```bash
git clone https://github.com/ranjangogoi61/scripttruth.git
cd scripttruth
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):

```
GEMINI_API_KEY=your_key_here
PARALLEL_API_KEY=your_key_here
```

- Gemini API key: https://aistudio.google.com (free tier, no card required)
- Parallel API key: https://platform.parallel.ai

Run it:

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

### API

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"scene_text": "Kolkata, 1987. Rahul scrolls Instagram on his phone.", "genre_mode": "modern"}'
```

---

## Try it

Paste a scene with a deliberate error, for example:

> *Kolkata, 1987. Rahul sits at a chai stall, scrolling Instagram on his phone.*

ScriptTruth flags the anachronism and links to a source establishing Instagram's 2010 launch.

**Note on the hosted demo:** the app runs on Render's free tier and may take 30–60 seconds to respond on the first request after a period of inactivity. It also uses the free tier of the Gemini API, which has a daily request quota — if verification reports a quota message, it resets every 24 hours.

---

## Reliability engineering

The system is built around the assumption that things fail. Handled explicitly:

- **Absence of evidence** never becomes a "contradicted" verdict
- **Citation validation** — a cited URL must exist in what was actually retrieved; hallucinated citations are rejected in code, not trusted to prompting
- **Model fallback chain** — free-tier quotas are per-model, so the app falls through a chain of models rather than failing when one is exhausted
- **Batched comparison** — all claims verified in a single call, cutting API usage from 7 requests per scene to 2
- **Selective retry** — transient errors (503, burst 429) retry with exponential backoff; permanent ones (404, daily quota) fail fast instead of wasting quota
- **Per-claim isolation** — one claim failing never kills the rest of the request
- **Timeouts and a global deadline** — partial results are returned rather than hanging
- **Prompt injection guard** — scene text is framed strictly as data, never as instructions
- **Deterministic behaviour** — `temperature=0.1`, and all scoring computed in Python rather than by the model

---

## Findings and learnings

**A model cannot verify its own claims.** The most important architectural insight is that generation and verification are different operations. A language model asked "is this period-accurate?" produces confident, fluent, sometimes invented answers. Only retrieval provides ground truth. This is why the search integration is structural rather than decorative.

**Absence of evidence is the hardest case.** The first instinct is to treat "no search results" as "this is wrong." For screenplays that is catastrophically wrong — fiction is full of invented brands, places, and people that correctly have no web presence. Distinguishing "I found evidence this is false" from "I found nothing" is essential.

**Free-tier quotas shape architecture.** Discovering mid-build that the daily quota was 20 requests forced a redesign from per-claim comparison calls to a single batched call. That constraint produced a better design — fewer round trips, faster responses, and lower cost at any scale.

**Reliability fixes can fight each other.** An early build combined sequential processing, a rate-limit stagger, and blind retries. Each was individually sensible; together they produced a 33-second response. Running claims concurrently made the other protections nearly free.

---

## The engine generalises

The pipeline — extract claims, retrieve evidence per claim, compare, cite — is domain-agnostic. Only the extraction prompt and output vocabulary are specific to screenplays. The same engine applies directly to newsroom claim verification, advertising claim substantiation, and podcast guest-claim checking. Screenplays are the first domain, not the only one.

---
## 🎥 Live Demo

Watch ScriptTruth in action as it detects historical inaccuracies, technology anachronisms, and fictional entities using Google Gemini and Parallel Search.

[![ScriptTruth Demo](https://img.youtube.com/vi/GbEfXByKoY0/maxresdefault.jpg)](https://www.youtube.com/watch?v=GbEfXByKoY0)

## License

MIT — see [LICENSE](LICENSE).

