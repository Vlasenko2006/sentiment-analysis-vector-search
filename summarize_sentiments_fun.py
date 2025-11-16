#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Summary Generator Function
Function-based version for integration into main pipeline

Created on November 3, 2025
@author: andreyvlasenko
"""

import os
import json
import requests
from datetime import datetime


def read_representatives_json(sentiment_analysis_dir, sentiment_type):
    """Read representative comments from JSON file"""
    json_file = os.path.join(
        sentiment_analysis_dir, 
        sentiment_type, 
        f'{sentiment_type}_representatives.json'
    )
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} {sentiment_type} representative comments")
        return data
    except Exception as e:
        print(f"❌ Error reading {json_file}: {e}")
        return None


def create_summary_prompt(comments, sentiment_type):
    """Create a prompt for LLM to summarize comments"""
    comment_texts = []
    for i, comment in enumerate(comments, 1):
        text = comment.get('text', '')
        confidence = comment.get('confidence', 0)
        comment_texts.append(f"{i}. [Confidence: {confidence:.2f}] {text}")
    
    combined_comments = "\n\n".join(comment_texts)
    
    prompt = f"""Analyze the following {sentiment_type.upper()} comments from customer reviews and provide a concise summary in EXACTLY 2-3 sentences.

{sentiment_type.upper()} COMMENTS:
{combined_comments}

Write a brief summary (2-3 sentences ONLY) explaining what aspects the commenters found {sentiment_type}. Focus on the main themes and common patterns.

Summary:"""
    
    return prompt


def query_groq_api(prompt, groq_api_key):
    """Query Groq API"""
    if not groq_api_key or groq_api_key == 'your-api-key-here':
        print("❌ GROQ_API_KEY not set")
        print("   Get free API key from: https://console.groq.com")
        return None
    
    try:
        print("🤖 Querying Groq API (llama-3.1-8b-instant)...")
        
        # Limit prompt length
        max_prompt_length = 6000
        if len(prompt) > max_prompt_length:
            print(f"   ⚠️  Prompt too long ({len(prompt)} chars), truncating to {max_prompt_length}")
            prompt = prompt[:max_prompt_length] + "\n\nSummary:"
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"❌ Groq API error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {error_detail}")
            except:
                print(f"   Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ Error querying Groq: {e}")
        return None


def save_summary(summary, sentiment_type, comments_data, sentiment_analysis_dir, llm_method):
    """Save summary to both TXT and JSON files"""
    output_dir = os.path.join(sentiment_analysis_dir, sentiment_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as TXT
    txt_file = os.path.join(output_dir, f'{sentiment_type}_summary.txt')
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{sentiment_type.upper()} SENTIMENT SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Based on: {len(comments_data)} representative comments\n")
            f.write(f"LLM: {llm_method}\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary)
            f.write("\n")
        print(f"✅ Saved summary to: {txt_file}")
    except Exception as e:
        print(f"❌ Error saving TXT: {e}")
    
    # Save as JSON
    json_file = os.path.join(output_dir, f'{sentiment_type}_summary.json')
    try:
        data = {
            "sentiment_type": sentiment_type.upper(),
            "generated_timestamp": datetime.now().isoformat(),
            "num_comments_analyzed": len(comments_data),
            "summary": summary,
            "llm_method": llm_method,
            "model_used": "llama-3.1-8b-instant" if llm_method == "groq" else llm_method
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved summary to: {json_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def process_sentiment_summary(sentiment_type, sentiment_analysis_dir, groq_api_key, llm_method):
    """Process and generate summary for a sentiment type"""
    print("\n" + "=" * 80)
    print(f"PROCESSING {sentiment_type.upper()} SENTIMENT SUMMARY")
    print("=" * 80)
    
    # Read representative comments
    comments = read_representatives_json(sentiment_analysis_dir, sentiment_type)
    if not comments:
        return False
    
    # Create prompt
    prompt = create_summary_prompt(comments, sentiment_type)
    print(f"\n📝 Created prompt with {len(comments)} comments")
    
    # Query LLM
    if llm_method == "groq":
        summary = query_groq_api(prompt, groq_api_key)
    else:
        print(f"❌ Unknown LLM method: {llm_method}")
        return False
    
    if not summary:
        print(f"❌ Failed to generate summary for {sentiment_type}")
        return False
    
    print(f"\n✅ Generated summary ({len(summary)} characters)")
    print("\n" + "-" * 80)
    print("SUMMARY PREVIEW:")
    print("-" * 80)
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    print("-" * 80)
    
    # Save summary
    save_summary(summary, sentiment_type, comments, sentiment_analysis_dir, llm_method)
    
    return True


def summarize_sentiments_fun(sentiment_analysis_dir, groq_api_key, llm_method="groq"):
    """
    Main function to generate AI summaries for sentiment analysis
    
    Args:
        sentiment_analysis_dir: Directory containing sentiment analysis results
        groq_api_key: Groq API key for LLM queries
        llm_method: LLM method to use (default: "groq")
    
    Returns:
        Dictionary with success status for each sentiment
    """
    print("=" * 80)
    print("SENTIMENT SUMMARY GENERATOR")
    print("=" * 80)
    
    print(f"\n📂 Sentiment analysis directory: {sentiment_analysis_dir}")
    print(f"🤖 Using LLM method: {llm_method}")
    
    # Check API key
    if llm_method == "groq":
        if not groq_api_key or groq_api_key == 'your-api-key-here':
            print("⚠️  Warning: GROQ_API_KEY not set")
            print("   Get free key from: https://console.groq.com")
            print("   Set environment variable: export GROQ_API_KEY='your-key-here'")
            return {"positive": False, "negative": False, "neutral": False}
    
    # Process each sentiment
    results = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        results[sentiment] = process_sentiment_summary(
            sentiment, sentiment_analysis_dir, groq_api_key, llm_method
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY GENERATION COMPLETE")
    print("=" * 80)
    
    for sentiment, success in results.items():
        if success:
            print(f"✅ {sentiment.capitalize()} sentiment summary generated")
        else:
            print(f"❌ Failed to generate {sentiment} sentiment summary")
    
    print(f"\n📁 Output location: {sentiment_analysis_dir}")
    print("   Check positive/, negative/, neutral/ folders for:")
    print("   - *_summary.txt (readable summary)")
    print("   - *_summary.json (structured data)")
    
    print("\n" + "=" * 80)
    
    return results
