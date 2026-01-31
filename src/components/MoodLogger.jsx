import React, { useState } from 'react';
import { Smile, Zap, Brain, Utensils, Save, CheckCircle } from 'lucide-react';

const MoodLogger = ({ onEntryLogged }) => {
    const [formData, setFormData] = useState({
        mood_score: 3,
        energy_level: 3,
        focus_level: 3,
        food_eaten: '',
        notes: ''
    });
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setSuccess(false);

        try {
            const response = await fetch('http://localhost:8000/log_entry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to log entry');
            }

            setSuccess(true);
            setFormData({
                mood_score: 3,
                energy_level: 3,
                focus_level: 3,
                food_eaten: '',
                notes: ''
            });

            console.log('Entry logged, refreshing dashboard...');
            if (onEntryLogged) {
                await onEntryLogged();
                console.log('Dashboard refreshed!');
            }

            setTimeout(() => setSuccess(false), 3000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const ScoreSelector = ({ label, icon: Icon, value, onChange, color, bgColor }) => (
        <div className="mb-5">
            <div className="flex items-center gap-2 mb-2">
                <Icon size={18} className={color} />
                <span className="font-medium text-gray-700">{label}</span>
            </div>
            <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map(score => (
                    <button
                        key={score}
                        type="button"
                        className={`w-10 h-10 rounded-lg font-semibold transition-all duration-200 ${value === score
                                ? `${bgColor} text-white shadow-md scale-105`
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }`}
                        onClick={() => onChange(score)}
                    >
                        {score}
                    </button>
                ))}
            </div>
        </div>
    );

    return (
        <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-5 flex items-center gap-2">
                📝 Log Today's Wellness
            </h3>

            {success && (
                <div className="flex items-center gap-2 px-4 py-3 bg-green-100 text-green-700 rounded-lg mb-4 animate-fadeIn">
                    <CheckCircle size={18} />
                    Entry logged successfully!
                </div>
            )}

            {error && (
                <div className="px-4 py-3 bg-red-100 text-red-700 rounded-lg mb-4">
                    {error}
                </div>
            )}

            <form onSubmit={handleSubmit}>
                <ScoreSelector
                    label="Mood Level"
                    icon={Smile}
                    value={formData.mood_score}
                    onChange={(val) => setFormData(p => ({ ...p, mood_score: val }))}
                    color="text-green-500"
                    bgColor="bg-green-500"
                />

                <ScoreSelector
                    label="Energy Level"
                    icon={Zap}
                    value={formData.energy_level}
                    onChange={(val) => setFormData(p => ({ ...p, energy_level: val }))}
                    color="text-orange-500"
                    bgColor="bg-orange-500"
                />

                <ScoreSelector
                    label="Focus Level"
                    icon={Brain}
                    value={formData.focus_level}
                    onChange={(val) => setFormData(p => ({ ...p, focus_level: val }))}
                    color="text-blue-500"
                    bgColor="bg-blue-500"
                />

                <div className="mb-4">
                    <label className="flex items-center gap-2 font-medium text-gray-700 mb-2">
                        <Utensils size={16} className="text-gray-500" />
                        What did you eat today?
                    </label>
                    <textarea
                        className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all resize-none text-sm"
                        placeholder="e.g., Oatmeal for breakfast, salad for lunch..."
                        value={formData.food_eaten}
                        onChange={(e) => setFormData(p => ({ ...p, food_eaten: e.target.value }))}
                        rows={2}
                    />
                </div>

                <div className="mb-5">
                    <label className="block font-medium text-gray-700 mb-2">Notes (optional)</label>
                    <input
                        type="text"
                        className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all text-sm"
                        placeholder="Any additional notes..."
                        value={formData.notes}
                        onChange={(e) => setFormData(p => ({ ...p, notes: e.target.value }))}
                    />
                </div>

                <button
                    type="submit"
                    className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-lg shadow-md hover:shadow-lg hover:from-green-600 hover:to-green-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={loading}
                >
                    <Save size={18} />
                    {loading ? 'Saving...' : 'Log Entry'}
                </button>
            </form>
        </div>
    );
};

export default MoodLogger;
