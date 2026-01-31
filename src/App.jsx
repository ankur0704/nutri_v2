import { useState } from 'react';
import Dashboard from './components/Dashboard';

function App() {
  const [view, setView] = useState('advisor');
  const [formData, setFormData] = useState({
    text: '',
    age: 25,
    goal: 'Maintain Weight'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'age' ? parseInt(value) || '' : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/recommend_diet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get recommendation.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      if (err.message.includes('Failed to fetch')) {
        setError('Network Error: Could not connect to backend (is it running on port 8000?)');
      } else {
        setError(err.message || 'An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 to-green-50 font-sans">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="flex flex-wrap justify-between items-center gap-4 mb-2">
            <h1 className="text-3xl font-bold text-gray-800" style={{ fontFamily: "'Red Rose', cursive" }}>🥗 NutriMate AI</h1>
            <nav className="flex gap-1 bg-white/50 p-1 rounded-full backdrop-blur-sm">
              <button
                className={`px-5 py-2 rounded-full font-medium transition-all duration-300 ${view === 'advisor'
                  ? 'bg-green-500 text-white shadow-md'
                  : 'text-gray-600 hover:bg-white/50'
                  }`}
                onClick={() => setView('advisor')}
              >
                Meal Advisor
              </button>
              <button
                className={`px-5 py-2 rounded-full font-medium transition-all duration-300 ${view === 'dashboard'
                  ? 'bg-green-500 text-white shadow-md'
                  : 'text-gray-600 hover:bg-white/50'
                  }`}
                onClick={() => setView('dashboard')}
              >
                My Analytics
              </button>
            </nav>
          </div>
          <p className="text-gray-500 text-sm">Emotion-Aware Personalized Diet Recommendation System</p>
        </header>

        {/* Main Content */}
        <main>
          {view === 'advisor' ? (
            <div className="animate-fadeIn">
              {/* Form Card */}
              <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
                <form onSubmit={handleSubmit}>
                  <div className="mb-6">
                    <label htmlFor="text" className="block font-semibold text-gray-700 mb-2">
                      How are you feeling today?
                    </label>
                    <textarea
                      id="text"
                      name="text"
                      className="w-full p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all resize-none"
                      placeholder="Describe your current state... (e.g., I feel stressed and have a lot of work)"
                      rows={4}
                      value={formData.text}
                      onChange={handleChange}
                      required
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div>
                      <label htmlFor="age" className="block font-semibold text-gray-700 mb-2">
                        Age
                      </label>
                      <input
                        type="number"
                        id="age"
                        name="age"
                        className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all"
                        value={formData.age}
                        onChange={handleChange}
                        min="1"
                        max="120"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="goal" className="block font-semibold text-gray-700 mb-2">
                        Goal
                      </label>
                      <select
                        id="goal"
                        name="goal"
                        className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all bg-white cursor-pointer"
                        value={formData.goal}
                        onChange={handleChange}
                      >
                        <option value="Maintain Weight">Maintain Weight</option>
                        <option value="Weight Loss">Weight Loss</option>
                        <option value="Weight Gain">Weight Gain</option>
                      </select>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="w-full py-4 bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={loading}
                  >
                    {loading ? 'Analyzing...' : 'Get Diet Recommendation'}
                  </button>
                </form>
              </div>

              {/* Loading State */}
              {loading && (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="w-12 h-12 border-4 border-green-200 border-t-green-500 rounded-full animate-spin mb-4"></div>
                  <p className="text-gray-500 font-medium">Analyzing your emotion...</p>
                </div>
              )}

              {/* Error State */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-6">
                  {error}
                </div>
              )}

              {/* Result Card */}
              {result && (
                <div className="bg-white rounded-2xl shadow-lg p-8 animate-fadeIn">
                  <div className="flex flex-wrap justify-between items-center gap-4 mb-6 pb-4 border-b border-gray-100">
                    <h2 className="text-xl font-bold text-gray-800">Analysis Result</h2>
                    {result.emotion && (
                      <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-full">
                        <span className="text-sm text-gray-600">Detected Emotion:</span>
                        <strong className="text-green-600">{result.emotion}</strong>
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl p-5 text-center">
                      <span className="text-3xl mb-2 block">🍳</span>
                      <div className="font-semibold text-gray-700 mb-1">Breakfast</div>
                      <div className="text-gray-600 text-sm">{result.diet_plan?.breakfast || 'N/A'}</div>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-5 text-center">
                      <span className="text-3xl mb-2 block">🥗</span>
                      <div className="font-semibold text-gray-700 mb-1">Lunch</div>
                      <div className="text-gray-600 text-sm">{result.diet_plan?.lunch || 'N/A'}</div>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-violet-50 rounded-xl p-5 text-center">
                      <span className="text-3xl mb-2 block">🍽️</span>
                      <div className="font-semibold text-gray-700 mb-1">Dinner</div>
                      <div className="text-gray-600 text-sm">{result.diet_plan?.dinner || 'N/A'}</div>
                    </div>
                  </div>

                  {result.notes && (
                    <div className="bg-amber-50 border-l-4 border-amber-400 rounded-r-xl p-4">
                      <div className="font-semibold text-amber-800 mb-2">💡 Why this recommendation?</div>
                      <p className="text-gray-600 text-sm">{result.notes}</p>
                    </div>
                  )}
                </div>
              )}
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
