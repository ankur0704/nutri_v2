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

## 🎓 Academic Uniqueness & Research Directions

> **For BE Computer Engineering Final Year Project**
>
> This section outlines how to elevate NutriMate AI from a standard web application to a **research-worthy, academically unique project** suitable for a Bachelor of Engineering degree.

### 🔬 Current State vs. Academic Requirements

| Aspect | Current Implementation | Required for Academic Uniqueness |
|--------|----------------------|----------------------------------|
| AI Model | Using pre-trained Gemini API | Build/fine-tune custom model |
| Data | Generic recommendations | Personalized learning from user data |
| Research | None | Published algorithm/approach |
| Novelty | Standard CRUD app | Novel contribution to the field |

---

### 🧠 Building Your Own ML Model (Critical for Uniqueness)

#### Option 1: Emotion Detection Model (NLP)
Build a custom emotion classification model instead of relying on Gemini:

```python
# Train on emotion datasets like:
# - GoEmotions (Google) - 58K Reddit comments, 27 emotions
# - ISEAR - International Survey on Emotion Antecedents
# - AffectiveText - SemEval dataset

from transformers import BertForSequenceClassification, Trainer

# Fine-tune BERT for emotion classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=6  # happy, sad, angry, anxious, stressed, neutral
)
```

**Research Contribution**: Novel fine-tuning approach for food-related emotional context.

#### Option 2: Food-Mood Correlation Model
Train a model that learns correlations between food intake and mood changes:

```python
# Features: nutritional content, meal timing, user history
# Target: mood score changes over time

from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras import Sequential, layers

# Custom neural network for mood prediction
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # Predicted mood score
])
```

**Research Contribution**: Personalized food-mood mapping using collaborative filtering.

#### Option 3: Hybrid Recommendation System
Combine content-based and collaborative filtering:

```
User Profile + Food Nutrition Data + Historical Mood Data
                    ↓
           Hybrid Neural Network
                    ↓
        Personalized Meal Recommendations
```

---

### 📊 Novel Features to Implement

| Feature | Description | Academic Value |
|---------|-------------|----------------|
| **Temporal Pattern Analysis** | Detect mood patterns across time (weekly, monthly cycles) | Time-series analysis research |
| **Gut-Brain Axis Modeling** | Model the connection between gut microbiome and mental health | Bioinformatics integration |
| **Personalized Nutrient Mapping** | Learn which specific nutrients affect individual users | Personalization algorithms |
| **Anomaly Detection** | Alert users to unusual mood-food patterns | Unsupervised learning |
| **Explainable AI (XAI)** | Show WHY certain foods are recommended | Transparency in AI |

---

### 🔄 Proposed System Architecture (Research-Grade)

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (React)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (FastAPI)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   EMOTION     │     │  RECOMMENDATION  │     │   ANALYTICS     │
│   DETECTOR    │     │     ENGINE       │     │    ENGINE       │
│  (Custom NLP) │     │ (Hybrid Model)   │     │ (Time-Series)   │
└───────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (MongoDB)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ User Profiles│  │  Food Data   │  │ Mood History │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

### 📝 Research Paper Topics

Consider writing a research paper on one of these topics:

1. **"Personalized Nutrition Recommendations Using Emotion-Aware Machine Learning"**
   - Focus: How emotional context improves diet recommendations

2. **"A Hybrid Deep Learning Approach for Food-Mood Correlation Analysis"**
   - Focus: Novel model architecture combining NLP and recommendation systems

3. **"Explainable AI in Health Tech: Making Nutrition Recommendations Transparent"**
   - Focus: XAI techniques applied to health recommendations

4. **"Real-time Mood Tracking and Dietary Intervention: A Mobile Health Approach"**
   - Focus: System design and user study results

---

### 🛠️ Technical Differentiators to Implement

| Component | Standard Approach | Your Unique Approach |
|-----------|------------------|---------------------|
| Emotion Analysis | API call to ChatGPT/Gemini | Custom BERT fine-tuned on food-emotion dataset |
| Recommendations | Rule-based matching | ML model trained on user feedback loop |
| Data Collection | Manual forms | Passive sensing (optional: wearable integration) |
| Personalization | Same for all users | Adapts to individual patterns over 2+ weeks |
| Validation | Demo only | A/B testing with real users, statistical analysis |

---

### 📈 Suggested Implementation Phases

#### Phase 3: Custom Model Development (4-6 weeks)
- [ ] Collect/curate emotion-food dataset
- [ ] Train custom emotion classifier (BERT/DistilBERT)
- [ ] Implement food-mood correlation model
- [ ] A/B test custom model vs Gemini API

#### Phase 4: Research & Validation (3-4 weeks)
- [ ] Conduct user study (minimum 20-30 participants)
- [ ] Collect 2 weeks of longitudinal data
- [ ] Statistical analysis of mood improvements
- [ ] Write research paper/project report

#### Phase 5: Advanced Features (2-3 weeks)
- [ ] Implement explainable AI (show reasoning)
- [ ] Add wearable device integration (optional)
- [ ] Deploy to production (Vercel + Railway)

---

### 📚 Datasets for Training Your Model

| Dataset | Description | Use Case |
|---------|-------------|----------|
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | 58K Reddit comments, 27 emotions | Emotion classification |
| [Food.com Recipes](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) | 230K recipes with ratings | Food recommendation |
| [USDA FoodData](https://fdc.nal.usda.gov/) | Nutritional content database | Nutrient analysis |
| [MyFitnessPal Data](https://www.kaggle.com/datasets) | User food logs | Pattern analysis |

---

### 🎯 What Makes This Project Unique for Evaluation

1. **Novel Algorithm**: Custom emotion-aware recommendation algorithm
2. **End-to-End System**: From data collection to actionable insights
3. **Real User Validation**: Statistical proof of effectiveness
4. **Publishable Research**: Paper-worthy contribution
5. **Technical Depth**: ML, NLP, full-stack, database, and analytics

---

### 💡 Quick Wins for Immediate Differentiation

If time is limited, focus on these high-impact additions:

1. **Custom Emotion Classifier** - Use Hugging Face to fine-tune a small model
2. **User Feedback Loop** - Let users rate recommendations to improve the model
3. **Explainability** - Show "why" behind each recommendation
4. **Statistical Dashboard** - Add p-values and confidence intervals to analytics
5. **Comparative Study** - Compare your approach vs. baseline in report

---

<p align="center">
  Made with 💚 by NutriMate AI Team
</p>