import React from 'react';

const EMOTION_EMOJIS = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  anxious: '😰',
  stressed: '😫',
  neutral: '😐',
};

const EmotionCard = ({ emotionAnalysis, loading }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 animate-fadeIn">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-2xl">🧠</span>
          <div>
            <h3 className="font-bold text-gray-800">Custom AI Model</h3>
            <p className="text-xs text-gray-500">Analyzing emotion...</p>
          </div>
        </div>
        <div className="flex justify-center py-4">
          <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (!emotionAnalysis) return null;

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 animate-fadeIn">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-2xl">🧠</span>
        <div>
          <h3 className="font-bold text-gray-800">Custom AI Model</h3>
          <p className="text-xs text-gray-500">DistilBERT – Trained on 58K samples</p>
        </div>
        <span className="ml-auto px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
          ~50ms
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        {/* Main detected emotion */}
        <div className="flex-1 min-w-[200px]">
          <div className="flex items-center gap-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
            <span className="text-3xl">
              {EMOTION_EMOJIS[emotionAnalysis.emotion] ?? '🤔'}
            </span>
            <div>
              <div className="text-lg font-bold text-gray-800 capitalize">
                {emotionAnalysis.emotion}
              </div>
              <div className="text-sm text-gray-500">
                Confidence: {(emotionAnalysis.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        {/* Top 3 emotion score bars */}
        <div className="flex-1 min-w-[200px]">
          <div className="text-xs text-gray-500 mb-2">All Emotions</div>
          <div className="space-y-1">
            {emotionAnalysis.all_scores &&
              Object.entries(emotionAnalysis.all_scores)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([emotion, score]) => (
                  <div key={emotion} className="flex items-center gap-2">
                    <span className="text-xs w-16 text-gray-600 capitalize">{emotion}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 w-10">
                      {(score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionCard;
