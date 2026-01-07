# NutriMate AI 

Emotion-Aware Personalized Diet Recommendation System. A Phase-1 project demo.

## Tech Stack
- **Frontend**: React (Vite)
- **Backend**: Python (FastAPI)
- **Styling**: Vanilla CSS (Modern Healthcare Theme)

## How to Run

### 1. Frontend Setup
The frontend runs on port `5173`.

1. Open a terminal.
2. Navigate to the project directory:
   ```bash
   cd "nutrition project"
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Open [http://localhost:5173](http://localhost:5173) in your browser.

### 2. Backend Setup (Required)
The backend runs on port `8000` and is required for the application to work.

1. Open a **new** terminal window.
2. Navigate to the backend directory:
   ```bash
   cd "nutrition project/backend"
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
5. The API will be available at [http://localhost:8000](http://localhost:8000).

## Usage
1. Enter how you are feeling (e.g., "I feel stressed and tired").
2. Enter your age and select a goal.
3. Click "Get Diet Recommendation" to see the AI-generated diet plan.