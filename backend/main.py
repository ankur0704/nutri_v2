from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List

# Load environment variables
load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GENAI_API_KEY)

# Use the standard model
model = genai.GenerativeModel('gemini-2.5-flash')

# MongoDB Connection
MONGODB_URL = os.getenv("MONGODB_URL")
client = None
db = None

async def connect_to_mongo():
    global client, db
    if MONGODB_URL:
        try:
            client = AsyncIOMotorClient(MONGODB_URL)
            db = client.nutrimate
            # Test the connection
            await client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas!")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
    else:
        print("⚠️ MONGODB_URL not found in .env")

app = FastAPI()

# Startup event to connect to MongoDB
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

# Shutdown event to close MongoDB connection
@app.on_event("shutdown")
async def shutdown_db_client():
    global client
    if client:
        client.close()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    text: str
    age: int
    goal: str

class MoodEntry(BaseModel):
    mood_score: int  # 1-5
    energy_level: int  # 1-5
    focus_level: int  # 1-5
    food_eaten: Optional[str] = None
    notes: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "NutriMate AI Backend is Running", "db_status": "connected" if db else "not connected"}

@app.post("/recommend_diet")
async def recommend_diet(user_input: UserInput):
    try:
        # Construct the prompt
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

        Output Format:
        Return ONLY valid JSON with this exact structure (no markdown, no backticks):
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

        # Generate content
        response = model.generate_content(prompt)
        
        # Clean up response to ensure it's valid JSON
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        result = json.loads(response_text)
        return result

    except Exception as e:
        import traceback
        print("Detailed Error Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

@app.post("/log_entry")
async def log_entry(entry: MoodEntry):
    """Log a mood/wellness entry to the database"""
    print(f"DEBUG: Received entry: {entry}")
    print(f"DEBUG: db object: {db}")
    
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        document = {
            "mood_score": entry.mood_score,
            "energy_level": entry.energy_level,
            "focus_level": entry.focus_level,
            "food_eaten": entry.food_eaten,
            "notes": entry.notes,
            "timestamp": datetime.utcnow()
        }
        print(f"DEBUG: Inserting document: {document}")
        result = await db.mood_entries.insert_one(document)
        print(f"DEBUG: Insert successful, id: {result.inserted_id}")
        return {"message": "Entry logged successfully", "id": str(result.inserted_id)}
    except Exception as e:
        import traceback
        print("ERROR in log_entry:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to log entry: {str(e)}")

@app.get("/get_history")
async def get_history(limit: int = 7):
    """Get recent mood/wellness history"""
    print(f"DEBUG get_history: db = {db}")
    
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        cursor = db.mood_entries.find().sort("timestamp", -1).limit(limit)
        entries = []
        async for doc in cursor:
            timestamp_val = doc.get("timestamp")
            entries.append({
                "id": str(doc["_id"]),
                "mood_score": doc.get("mood_score"),
                "energy_level": doc.get("energy_level"),
                "focus_level": doc.get("focus_level"),
                "food_eaten": doc.get("food_eaten"),
                "notes": doc.get("notes"),
                "timestamp": timestamp_val.isoformat() if timestamp_val else None
            })
        print(f"DEBUG get_history: returning {len(entries)} entries")
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        import traceback
        print("ERROR in get_history:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/analytics")
async def get_analytics():
    """Get aggregated analytics for the dashboard"""
    print(f"DEBUG analytics: db = {db}")
    
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Get last 30 entries
        cursor = db.mood_entries.find().sort("timestamp", -1).limit(30)
        entries = []
        async for doc in cursor:
            entries.append(doc)
        
        print(f"DEBUG analytics: found {len(entries)} entries")
        
        if not entries:
            return {"message": "No data available", "has_data": False}
        
        # Calculate averages safely
        mood_scores = [e.get("mood_score", 0) or 0 for e in entries]
        energy_levels = [e.get("energy_level", 0) or 0 for e in entries]
        focus_levels = [e.get("focus_level", 0) or 0 for e in entries]
        
        avg_mood = sum(mood_scores) / len(entries)
        avg_energy = sum(energy_levels) / len(entries)
        avg_focus = sum(focus_levels) / len(entries)
        
        result = {
            "has_data": True,
            "total_entries": len(entries),
            "averages": {
                "mood": round(avg_mood, 1),
                "energy": round(avg_energy, 1),
                "focus": round(avg_focus, 1)
            }
        }
        print(f"DEBUG analytics: returning {result}")
        return result
    except Exception as e:
        import traceback
        print("ERROR in analytics:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
