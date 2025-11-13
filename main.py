import os
import re
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="WatchDog Backend", version="0.2.0")

# CORS for extension + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: Optional[str] = None


class AnalyzeTextResponse(BaseModel):
    flagged: bool
    label: str
    preview: str
    scores: Dict[str, float]
    topTerms: List[str]


# ---------- Simple heuristic aligned with content script ----------
WORDLIST: Dict[str, List[str]] = {
    "abuse": [
        "ganda", "bakwas", "bhosd", "gandu", "madarchod", "bhenchod", "kutte",
        "kameena", "harami", "bewakoof",
    ],
    "sexual": [
        "sexy", "nsfw", "nangi", "nude", "18+", "xxx", "rand", "randi", "chaddi",
        "breast", "boobs",
    ],
    "slur": [
        "chutiya", "kutia", "chakk", "bhangi", "bihari", "madrasi", "jhatu",
    ],
    "hinglish": [
        "bc", "mc", "bsdk", "chod", "lund", "tatti", "gaand", "kutti", "kuttiya",
    ],
}

WEIGHTS: Dict[str, float] = {"abuse": 1.0, "sexual": 1.2, "slur": 1.1, "hinglish": 0.7}
ABUSE_THRESHOLD = 1.0
HINGLISH_COMBO_THRESHOLD = 2  # sexual + hinglish hits >= 2 => flagged


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9+]+", text.lower())


def score_text(text: str) -> AnalyzeTextResponse:
    tokens = simple_tokenize(text)
    joined = " ".join(tokens)

    counter: Dict[str, int] = {k: 0 for k in WORDLIST}
    term_hits: Dict[str, int] = {}

    for cat, words in WORDLIST.items():
        for w in words:
            pattern = re.compile(rf"\b{re.escape(w.lower())}\b")
            matches = pattern.findall(joined)
            if matches:
                hit_count = len(matches)
                counter[cat] += hit_count
                term_hits[w] = term_hits.get(w, 0) + hit_count

    scores: Dict[str, float] = {cat: counter[cat] * WEIGHTS[cat] for cat in counter}
    top_cat = max(scores, key=scores.get) if scores else "abuse"
    top_score = scores.get(top_cat, 0.0)

    sexual_hinglish_combo = counter["sexual"] + counter["hinglish"]
    flagged = (top_score >= ABUSE_THRESHOLD) or (sexual_hinglish_combo >= HINGLISH_COMBO_THRESHOLD)
    label = "Flagged (18+)" if flagged else "Clear"

    sorted_terms = sorted(term_hits.items(), key=lambda x: (-x[1], x[0]))
    top_terms = [t for t, _ in sorted_terms[:5]]
    preview = text[:200]

    return AnalyzeTextResponse(
        flagged=flagged,
        label=label,
        preview=preview,
        scores=scores,
        topTerms=top_terms,
    )


# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.post("/api/analyze/text", response_model=AnalyzeTextResponse)
def analyze_text(req: AnalyzeTextRequest):
    return score_text(req.text)


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # pragma: no cover - best-effort diagnostics
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:  # pragma: no cover
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
