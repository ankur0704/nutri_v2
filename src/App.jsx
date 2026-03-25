import { useState } from 'react';
import Dashboard from './components/Dashboard';
import MealAdvisorForm from './components/MealAdvisorForm';
import EmotionCard from './components/EmotionCard';
import DietResultCard from './components/DietResultCard';

function App() {
  const [view, setView] = useState('advisor');
  const [result, setResult] = useState(null);
  const [emotionAnalysis, setEmotionAnalysis] = useState(null);
  const [emotionLoading, setEmotionLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 to-green-50 font-sans">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="flex flex-wrap justify-between items-center gap-4 mb-2">
            <h1
              className="text-3xl font-bold text-gray-800"
              style={{ fontFamily: "'Red Rose', cursive" }}
            >
              🥗 NutriMate AI
            </h1>
            <nav className="flex gap-1 bg-white/50 p-1 rounded-full backdrop-blur-sm">
              <button
                className={`px-5 py-2 rounded-full font-medium transition-all duration-300 ${
                  view === 'advisor'
                    ? 'bg-green-500 text-white shadow-md'
                    : 'text-gray-600 hover:bg-white/50'
                }`}
                onClick={() => setView('advisor')}
              >
                Meal Advisor
              </button>
              <button
                className={`px-5 py-2 rounded-full font-medium transition-all duration-300 ${
                  view === 'dashboard'
                    ? 'bg-green-500 text-white shadow-md'
                    : 'text-gray-600 hover:bg-white/50'
                }`}
                onClick={() => setView('dashboard')}
              >
                My Analytics
              </button>
            </nav>
          </div>
          <p className="text-gray-500 text-sm">
            Emotion-Aware Personalized Diet Recommendation System
          </p>
        </header>

        {/* Main Content */}
        <main>
          {view === 'advisor' ? (
            <div className="animate-fadeIn">
              <MealAdvisorForm
                onResult={setResult}
                onEmotionResult={(data) => {
                  setEmotionAnalysis(data);
                  setEmotionLoading(false);
                }}
                onEmotionStart={() => {
                  setEmotionLoading(true);
                  setEmotionAnalysis(null);
                }}
              />
              <EmotionCard emotionAnalysis={emotionAnalysis} loading={emotionLoading} />
              <DietResultCard result={result} emotionAnalysis={emotionAnalysis} />
            </div>
          ) : (
            <Dashboard />
          )}
        </main>

        {/* Footer */}
        <footer className="text-center mt-12 py-6 text-gray-400 text-sm border-t border-gray-200">
          <p>NutriMate AI – Phase 2 Analytics | Advanced Engineering Project</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
