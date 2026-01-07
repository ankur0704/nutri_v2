from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    text: str
    age: int
    goal: str

@app.get("/")
def read_root():
    return {"message": "NutriMate AI Backend is Running"}

@app.post("/recommend_diet")
def recommend_diet(user_input: UserInput):
    # Mock AI Logic for Demo
    text_lower = user_input.text.lower()
    
    # Simple rule-based emotion detection
    emotion = "Neutral"
    if any(word in text_lower for word in ["stress", "anxi", "worr", "tired", "exhaust"]):
        emotion = "Stressed/Tired"
    elif any(word in text_lower for word in ["happ", "good", "great", "excit"]):
        emotion = "Happy/Energetic"
    elif any(word in text_lower for word in ["sad", "down", "depress", "blue"]):
        emotion = "Sad/Low"
    elif any(word in text_lower for word in ["angry", "mad", "upse"]):
        emotion = "Angry/Frustrated"

    # Diet Plan Logic based on Goal & Emotion
    # This is where a real AI model would generate content
    diet_plan = {}
    notes = ""

    if emotion == "Stressed/Tired":
        diet_plan = {
            "breakfast": "Oatmeal with almonds and blueberries (Magnesium boost)",
            "lunch": "Grilled chicken salad with spinach and avocado",
            "dinner": "Salmon with quinoa and steamed broccoli",
        }
        notes = "When stressed, your body depletes magnesium. We've included nuts, seeds, and leafy greens to help replenish it and lower cortisol levels."
    
    elif emotion == "Sad/Low":
        diet_plan = {
            "breakfast": "Greek yogurt with honey and walnuts",
            "lunch": "Turkey wrap with whole wheat tortilla",
            "dinner": "Dark chocolate (small piece) and berry smoothie bowl",
        }
        notes = "To boost serotonin, we recommended foods rich in Tryptophan (turkey, yogurt) and antioxidants."

    elif "Weight Loss" in user_input.goal: # Fallback based on goal if emotion is neutral/other
        diet_plan = {
            "breakfast": "Green smoothie with protein powder",
            "lunch": "Lentil soup with mixed vegetables",
            "dinner": "Grilled fish with asparagus",
        }
        notes = "Focusing on high protein and fiber to keep you full longer while maintaining a calorie deficit."

    else: # Default/Maintenance
        diet_plan = {
            "breakfast": "Scrambled eggs with toast",
            "lunch": "Chicken sandwich with side salad",
            "dinner": "Stir-fry tofu with brown rice",
        }
        notes = "A balanced plan to maintain your current weight and energy levels."

    return {
        "emotion": emotion,
        "diet_plan": diet_plan,
        "notes": notes
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
