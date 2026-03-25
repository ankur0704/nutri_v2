import React from 'react';

const DietResultCard = ({ result, emotionAnalysis }) => {
  if (!result) return null;

  const emotionMismatch =
    emotionAnalysis &&
    result.emotion &&
    emotionAnalysis.emotion !== result.emotion.toLowerCase();

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4 mb-6 pb-4 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🤖</span>
          <div>
            <h2 className="text-xl font-bold text-gray-800">Gemini AI Analysis</h2>
            <p className="text-xs text-gray-500">Full diet recommendation</p>
          </div>
        </div>
        {result.emotion && (
          <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-full">
            <span className="text-sm text-gray-600">Gemini Emotion:</span>
            <strong className="text-green-600">{result.emotion}</strong>
          </div>
        )}
      </div>

      {/* Model comparison alert */}
      {emotionMismatch && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6">
          <span className="font-medium text-amber-800">🔄 Model Comparison: </span>
          <span className="text-amber-700">
            Custom AI detected <strong className="capitalize">{emotionAnalysis.emotion}</strong>
            {' '}while Gemini detected <strong>{result.emotion}</strong>.
            Both interpretations are valid!
          </span>
        </div>
      )}

      {/* Meal cards */}
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

      {/* Notes */}
      {result.notes && (
        <div className="bg-amber-50 border-l-4 border-amber-400 rounded-r-xl p-4">
          <div className="font-semibold text-amber-800 mb-2">💡 Why this recommendation?</div>
          <p className="text-gray-600 text-sm">{result.notes}</p>
        </div>
      )}
    </div>
  );
};

export default DietResultCard;
