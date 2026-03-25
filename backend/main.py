from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import json
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from typing import Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nutrimate")

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# ─── Gemini configuration ─────────────────────────────────────────────────────
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"},
)

# ─── MongoDB ──────────────────────────────────────────────────────────────────
MONGODB_URL = os.getenv("MONGODB_URL")

# ─── Custom Emotion Model ─────────────────────────────────────────────────────
EMOTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "emotion_model")
EMOTION_LABELS = ["angry", "anxious", "happy", "neutral", "sad", "stressed"]


def _load_emotion_model():
    """Load the custom trained emotion classifier. Returns (tokenizer, model, device) or (None, None, None)."""
    try:
        if os.path.exists(EMOTION_MODEL_PATH):
            logger.info("Loading custom emotion model...")
            tokenizer = DistilBertTokenizer.from_pretrained(EMOTION_MODEL_PATH)
            model = DistilBertForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info("Custom emotion model loaded on %s", device)
            return tokenizer, model, device
        else:
            logger.warning("Emotion model not found at %s", EMOTION_MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load emotion model: %s", exc)
    return None, None, None


async def _connect_to_mongo():
    """Connect to MongoDB and return client + db, or (None, None) on failure."""
    if not MONGODB_URL:
        logger.warning("MONGODB_URL not found in .env")
        return None, None
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        await client.admin.command("ping")
        logger.info("Connected to MongoDB Atlas!")
        return client, client.nutrimate
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        return None, None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    mongo_client, db = await _connect_to_mongo()
    tokenizer, emotion_model, emotion_device = _load_emotion_model()
    app.state.mongo_client = mongo_client
    app.state.db = db
    app.state.emotion_tokenizer = tokenizer
    app.state.emotion_model = emotion_model
    app.state.emotion_device = emotion_device
    yield
    # Shutdown
    if app.state.mongo_client:
        app.state.mongo_client.close()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

# CORS – restrict to known origins; expand for production
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ──────────────────────────────────────────────────────────
class UserInput(BaseModel):
    text: str = Field(..., min_length=2, max_length=2000)
    age: int = Field(..., ge=1, le=120)
    goal: str = Field(..., pattern="^(Maintain Weight|Weight Loss|Weight Gain)$")


class MoodEntry(BaseModel):
    mood_score: int = Field(..., ge=1, le=5)
    energy_level: int = Field(..., ge=1, le=5)
    focus_level: int = Field(..., ge=1, le=5)
    food_eaten: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=500)


class EmotionInput(BaseModel):
    text: str = Field(..., min_length=2, max_length=2000)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _predict_emotion(text: str, tokenizer, model, device) -> dict:
    """Run inference with the custom emotion model."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    all_scores = {
        label: round(probs[0][i].item(), 3)
        for i, label in enumerate(EMOTION_LABELS)
    }
    return {
        "emotion": EMOTION_LABELS[pred_idx],
        "confidence": round(confidence, 3),
        "all_scores": all_scores,
        "model": "custom_distilbert",
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def read_root(request: Request):
    return {
        "message": "NutriMate AI Backend is Running",
        "db_status": "connected" if request.app.state.db is not None else "not connected",
        "emotion_model": "loaded" if request.app.state.emotion_model is not None else "not loaded",
    }


@app.post("/analyze_emotion")
async def analyze_emotion(input_data: EmotionInput, request: Request):
    """Analyze emotion using custom trained DistilBERT model."""
    tokenizer = request.app.state.emotion_tokenizer
    model = request.app.state.emotion_model
    device = request.app.state.emotion_device

    if not model or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Emotion model not loaded. Train the model first.",
        )
    try:
        return _predict_emotion(input_data.text, tokenizer, model, device)
    except Exception as exc:
        logger.exception("Emotion analysis failed")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {exc}")


@app.post("/recommend_diet")
async def recommend_diet(user_input: UserInput):
    prompt = f"""
    Act as a professional nutritionist and empathetic AI advisor.

    User Profile:
    - Age: {user_input.age}
    - Goal: {user_input.goal}
    - Current State/Feeling: "{user_input.text}"

    Task:
    1. Analyze the user's emotion based on their input text.
    2. Create a one-day diet plan (Breakfast, Lunch, Dinner) specifically tailored to their emotion and goal.
       (e.g., if stressed, suggest calming foods like magnesium-rich items).
    3. Write a short, encouraging note explaining why you picked these foods.

    Return a JSON object with this exact structure:
    {{
        "emotion": "Detected Emotion",
        "diet_plan": {{
            "breakfast": "Meal details",
            "lunch": "Meal details",
            "dinner": "Meal details"
        }},
        "notes": "Explanation and motivation"
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        result = json.loads(response.text)
        return result
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s | Raw: %s", exc, response.text[:200])
        raise HTTPException(
            status_code=502,
            detail="AI returned an unexpected response format. Please try again.",
        )
    except Exception as exc:
        logger.exception("Diet recommendation failed")
        raise HTTPException(status_code=500, detail=f"Server Error: {exc}")


@app.post("/log_entry")
async def log_entry(entry: MoodEntry, request: Request):
    """Log a mood/wellness entry to the database."""
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        document = {
            "mood_score": entry.mood_score,
            "energy_level": entry.energy_level,
            "focus_level": entry.focus_level,
            "food_eaten": entry.food_eaten,
            "notes": entry.notes,
            "timestamp": datetime.now(timezone.utc),
        }
        result = await db.mood_entries.insert_one(document)
        logger.info("Logged mood entry id=%s", result.inserted_id)
        return {"message": "Entry logged successfully", "id": str(result.inserted_id)}
    except Exception as exc:
        logger.exception("Failed to log entry")
        raise HTTPException(status_code=500, detail=f"Failed to log entry: {exc}")


@app.get("/get_history")
async def get_history(request: Request, limit: int = 7):
    """Get recent mood/wellness history."""
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        cursor = db.mood_entries.find().sort("timestamp", -1).limit(limit)
        entries = []
        async for doc in cursor:
            ts = doc.get("timestamp")
            entries.append({
                "id": str(doc["_id"]),
                "mood_score": doc.get("mood_score"),
                "energy_level": doc.get("energy_level"),
                "focus_level": doc.get("focus_level"),
                "food_eaten": doc.get("food_eaten"),
                "notes": doc.get("notes"),
                "timestamp": ts.isoformat() if ts else None,
            })
        logger.info("Returning %d history entries", len(entries))
        return {"entries": entries, "count": len(entries)}
    except Exception as exc:
        logger.exception("Failed to fetch history")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {exc}")


@app.get("/analytics")
async def get_analytics(request: Request):
    """Get aggregated analytics for the dashboard."""
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        cursor = db.mood_entries.find().sort("timestamp", -1).limit(30)
        entries = [doc async for doc in cursor]

        if not entries:
            return {"message": "No data available", "has_data": False}

        mood_scores = [e.get("mood_score") or 0 for e in entries]
        energy_levels = [e.get("energy_level") or 0 for e in entries]
        focus_levels = [e.get("focus_level") or 0 for e in entries]
        n = len(entries)

        result = {
            "has_data": True,
            "total_entries": n,
            "averages": {
                "mood": round(sum(mood_scores) / n, 1),
                "energy": round(sum(energy_levels) / n, 1),
                "focus": round(sum(focus_levels) / n, 1),
            },
        }
        logger.info("Analytics result: %s", result)
        return result
    except Exception as exc:
        logger.exception("Failed to get analytics")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
