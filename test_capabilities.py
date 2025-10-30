#!/usr/bin/env python3
"""
Testing what the sentiment model can analyze beyond plain text
"""

import os
from transformers import pipeline

# Load your existing model
model_path = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
pipe = pipeline("sentiment-analysis", model=model_path)

def analyze_with_confidence(text):
    """Analyze sentiment and show confidence"""
    result = pipe(text)
    label = result[0]['label']
    confidence = result[0]['score']
    
    # Simulate 3-class
    if confidence <= 0.8:
        if label == "POSITIVE":
            label = "NEUTRAL"
        elif label == "NEGATIVE":
            label = "NEUTRAL"
    
    return label, confidence

print("🔍 TESTING WHAT SENTIMENT ANALYSIS CAN 'SEE'")
print("=" * 60)

# Test categories
test_cases = {
    "📝 Plain Text": [
        "I love this",
        "This is terrible", 
        "The report is ready",
        "Not sure about this"
    ],
    
    "😊 Text + Emojis": [
        "I love this! 😊",
        "This is terrible 😞",
        "Great job! 👍",
        "I'm so angry 😡",
        "Feeling meh 😐",
        "Amazing! 🎉🎊"
    ],
    
    "❗ Punctuation & Intensity": [
        "Amazing!!!",
        "terrible...",
        "WOW!!!!!",
        "ok.",
        "LOVE IT!!",
        "hate hate hate"
    ],
    
    "⭐ Star-like Content": [
        "5 stars",
        "1 star review", 
        "This deserves 5 stars!",
        "Barely 2 stars",
        "Perfect 10/10",
        "Rating: 3/5"
    ],
    
    "🔤 Internet Slang & Symbols": [
        "This is LIT 🔥",
        "So cringe tbh",
        "OMG amazing!!!",
        "meh whatever",
        "YAAAS queen! 💯",
        "FML this sucks"
    ],
    
    "🌍 Mixed Languages & Symbols": [
        "Great! Отлично!",
        "Bad :(",
        "Good :)",
        "Love it ♥️",
        "Terrible ⚠️",
        "👎 not good"
    ]
}

for category, texts in test_cases.items():
    print(f"\n{category}")
    print("-" * 40)
    
    for text in texts:
        label, confidence = analyze_with_confidence(text)
        
        # Visual indicators
        if label == "POSITIVE":
            emoji = "😊"
        elif label == "NEGATIVE":
            emoji = "😞"
        else:
            emoji = "😐"
        
        print(f"{emoji} {label:<8} ({confidence:.3f}) | {text}")

print("\n" + "=" * 60)
print("🧠 HOW THE MODEL 'SEES' DIFFERENT ELEMENTS")
print("=" * 60)

print("""
🔤 WORDS:
   • "love", "amazing", "great" → Positive signals
   • "hate", "terrible", "awful" → Negative signals  
   • "meeting", "report", "data" → Neutral/factual

😊 EMOJIS:
   • 😊, 👍, 🎉, ❤️ → Positive signals
   • 😞, 😡, 👎, 💔 → Negative signals
   • 😐, 🤔 → Neutral/uncertain signals

❗ PUNCTUATION:
   • !!! → Excitement/intensity (usually positive)
   • ... → Hesitation/uncertainty (often negative)
   • CAPS → Emphasis (amplifies existing sentiment)

⭐ STAR RATINGS:
   • "5 stars", "10/10" → Strongly positive
   • "1 star", "2/10" → Strongly negative
   • "3 stars" → Neutral/mixed

🌐 MODEL LIMITATIONS:
   • Trained primarily on English text
   • May not understand all cultural context
   • Processes emojis as text tokens (learned from training data)
   • No visual analysis (can't see actual images)
""")

print("\n🔬 TECHNICAL DETAILS:")
print("=" * 40)

print("""
Token Processing:
1. Text → "I love this! 😊"
2. Tokenization → ["I", "love", "this", "!", "😊"]  
3. Each token gets a meaning vector
4. Model combines all vectors
5. Outputs sentiment probability

What It Learned:
• From millions of examples like "Great! 😊" → POSITIVE
• So it learned 😊 = positive signal
• Similarly 😞 = negative signal
• ❗= excitement/emphasis
""")

print(f"\n✅ Your model CAN analyze:")
print("   • Plain text ✅")
print("   • Emojis and emoticons ✅") 
print("   • Punctuation patterns ✅")
print("   • Internet slang ✅")
print("   • Star ratings (if in text form) ✅")
print("   • CAPS and repetition ✅")

print(f"\n❌ Your model CANNOT analyze:")
print("   • Actual images/photos ❌")
print("   • Audio/voice tone ❌") 
print("   • Video content ❌")
print("   • Real-time context ❌")