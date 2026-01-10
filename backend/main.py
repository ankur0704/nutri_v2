from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GENAI_API_KEY)

# Use the standard model
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow ALL origins for debugging
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
        # Remove potential markdown code blocks if Gemini adds them
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
