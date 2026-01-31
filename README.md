# 🥗 NutriMate AI

> **Emotion-Aware Personalized Diet Recommendation System**
> 
> An AI-powered wellness application that understands your emotional state and provides personalized diet recommendations to optimize your mental and physical health.

![NutriMate AI](https://img.shields.io/badge/Status-Active-brightgreen) ![React](https://img.shields.io/badge/React-18.x-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-darkgreen)

---

## ✨ Features

### 🧠 AI-Powered Diet Recommendations
- Analyzes your emotional input using Google Gemini AI
- Generates personalized meal plans based on mood, age, and health goals
- Provides nutritional insights and food suggestions

### 📊 Wellness Analytics Dashboard
- Track mood, energy, and focus levels over time
- Interactive charts visualizing your wellness trends
- Nutrient impact analysis showing how foods affect your mental state

### 📝 Mood & Wellness Logging
- Log daily mood, energy, and focus scores (1-5)
- Record food intake and personal notes
- All data persisted in MongoDB Atlas cloud database

### 📄 Clinical Report Export
- Generate professional PDF reports of your wellness data
- Includes mood trends, food correlations, and AI insights
- Perfect for sharing with healthcare providers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18 (Vite), Recharts, Lucide Icons |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI** | Google Gemini 2.0 Flash |
| **Database** | MongoDB Atlas (Cloud) |
| **Styling** | Vanilla CSS (Modern Healthcare Theme) |

---

## 📁 Project Structure

```
nutrition project/
├── backend/
│   ├── main.py              # FastAPI server with all endpoints
│   ├── requirements.txt     # Python dependencies
│   └── .env                  # Environment variables (API keys)
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx    # Analytics dashboard with charts
│   │   ├── MoodLogger.jsx   # Wellness logging form
│   │   ├── DietForm.jsx     # AI diet recommendation form
│   │   └── Navbar.jsx       # Navigation component
│   ├── App.jsx              # Main application component
│   ├── App.css              # Global styles
│   └── main.jsx             # React entry point
├── package.json             # Node.js dependencies
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.9+
- **MongoDB Atlas** account (free tier works)
- **Google AI API Key** (Gemini)

### 1️⃣ Clone & Install Frontend

```bash
# Navigate to project directory
cd "nutrition project"

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: **http://localhost:5173**

### 2️⃣ Setup Backend

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create or edit `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URL=mongodb+srv://username:password@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

> ⚠️ **Important**: URL-encode special characters in your MongoDB password (e.g., `@` becomes `%40`)

### 4️⃣ Start Backend Server

```bash
# From the backend directory
python main.py
```

Backend runs at: **http://localhost:8000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check & connection status |
| `POST` | `/recommend_diet` | Get AI diet recommendation |
| `POST` | `/log_entry` | Log a mood/wellness entry |
| `GET` | `/get_history` | Get recent wellness entries |
| `GET` | `/analytics` | Get aggregated analytics data |

### Example: Log a Wellness Entry

```bash
curl -X POST http://localhost:8000/log_entry \
  -H "Content-Type: application/json" \
  -d '{"mood_score": 4, "energy_level": 3, "focus_level": 4, "food_eaten": "Salmon salad", "notes": "Feeling good"}'
```

---

## 🖥️ Usage Guide

### Getting Diet Recommendations
1. Navigate to the **home page**
2. Enter how you're feeling (e.g., "I feel stressed and tired")
3. Enter your age and select a health goal
4. Click **"Get Diet Recommendation"**
5. View your personalized meal plan

### Logging Wellness Data
1. Go to **"My Analytics"** tab
2. Use the **"Log Today's Wellness"** form on the right
3. Select your mood, energy, and focus levels (1-5)
4. Add what you ate and any notes
5. Click **"Log Entry"**

### Viewing Analytics
1. Go to **"My Analytics"** tab
2. View your mood & focus trends chart
3. See nutrient impact analysis
4. Export a clinical PDF report

---

## 🔧 Development

### Running Both Servers

**Terminal 1 (Frontend):**
```bash
cd "nutrition project"
npm run dev
```

**Terminal 2 (Backend):**
```bash
cd "nutrition project/backend"
python main.py
```

### Hot Reload
- Frontend: Vite provides instant HMR
- Backend: Use `uvicorn main:app --reload` for auto-reload

---

## 🗄️ Database Schema

### Collection: `mood_entries`

```json
{
  "_id": "ObjectId",
  "mood_score": 1-5,
  "energy_level": 1-5,
  "focus_level": 1-5,
  "food_eaten": "string",
  "notes": "string",
  "timestamp": "ISODate"
}
```

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google AI Gemini API key | ✅ Yes |
| `MONGODB_URL` | MongoDB Atlas connection string | ✅ Yes |

---

## 📝 License

This project is for educational purposes.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

<p align="center">
  Made with 💚 by NutriMate AI Team
</p>