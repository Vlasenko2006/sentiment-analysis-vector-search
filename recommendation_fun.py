#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Generator Function
Generates actionable recommendations based on sentiment summaries

Created on November 3, 2025
@author: andreyvlasenko
"""

import os
import json
import requests
from datetime import datetime


def read_summary_file(sentiment_analysis_dir, sentiment_type):
    """Read summary text file for a sentiment type"""
    txt_file = os.path.join(
        sentiment_analysis_dir, 
        sentiment_type, 
        f'{sentiment_type}_summary.txt'
    )
    
    if not os.path.exists(txt_file):
        print(f"❌ File not found: {txt_file}")
        return None
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Loaded {sentiment_type} summary ({len(content)} characters)")
        return content
    except Exception as e:
        print(f"❌ Error reading {txt_file}: {e}")
        return None


def create_recommendation_prompt(prompt, positive_summary, negative_summary):
    """Create a prompt for LLM to generate recommendations"""
    
    full_prompt = f"""{prompt}

POSITIVE FEEDBACK SUMMARY:
{positive_summary}

NEGATIVE FEEDBACK SUMMARY:
{negative_summary}

Please provide 3 actionable recommendations:"""
    
    return full_prompt


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
            prompt = prompt[:max_prompt_length] + "\n\nPlease provide 3 actionable recommendations:"
        
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
                "max_tokens": 500
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


def save_recommendation(recommendation, sentiment_analysis_dir, llm_method):
    """Save recommendation to both TXT and JSON files"""
    output_dir = os.path.join(sentiment_analysis_dir, 'recommendation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as TXT
    txt_file = os.path.join(output_dir, 'recommendation.txt')
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LLM: {llm_method}\n")
            f.write("=" * 80 + "\n\n")
            f.write(recommendation)
            f.write("\n")
        print(f"✅ Saved recommendation to: {txt_file}")
    except Exception as e:
        print(f"❌ Error saving TXT: {e}")
    
    # Save as JSON
    json_file = os.path.join(output_dir, 'recommendation.json')
    try:
        data = {
            "generated_timestamp": datetime.now().isoformat(),
            "recommendation": recommendation,
            "llm_method": llm_method,
            "model_used": "llama-3.1-8b-instant" if llm_method == "groq" else llm_method
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved recommendation to: {json_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def recommendation_fun(prompt, sentiment_analysis_dir, groq_api_key, llm_method="groq"):
    """
    Main function to generate recommendations based on sentiment summaries
    
    Args:
        prompt: Custom prompt for generating recommendations
        sentiment_analysis_dir: Directory containing sentiment analysis results
        groq_api_key: Groq API key for LLM queries
        llm_method: LLM method to use (default: "groq")
    
    Returns:
        Boolean indicating success/failure
    """
    print("\n" + "=" * 80)
    print("RECOMMENDATION GENERATOR")
    print("=" * 80)
    
    print(f"\n📂 Sentiment analysis directory: {sentiment_analysis_dir}")
    print(f"🤖 Using LLM method: {llm_method}")
    
    # Check API key
    if llm_method == "groq":
        if not groq_api_key or groq_api_key == 'your-api-key-here':
            print("⚠️  Warning: GROQ_API_KEY not set")
            print("   Get free key from: https://console.groq.com")
            print("   Set environment variable: export GROQ_API_KEY='your-key-here'")
            return False
    
    # Read positive summary
    print("\n📖 Reading sentiment summaries...")
    positive_summary = read_summary_file(sentiment_analysis_dir, 'positive')
    if not positive_summary:
        print("❌ Failed to read positive summary")
        return False
    
    # Read negative summary
    negative_summary = read_summary_file(sentiment_analysis_dir, 'negative')
    if not negative_summary:
        print("❌ Failed to read negative summary")
        return False
    
    # Create recommendation prompt
    full_prompt = create_recommendation_prompt(prompt, positive_summary, negative_summary)
    print(f"\n📝 Created prompt ({len(full_prompt)} characters)")
    
    # Query LLM
    if llm_method == "groq":
        recommendation = query_groq_api(full_prompt, groq_api_key)
    else:
        print(f"❌ Unknown LLM method: {llm_method}")
        return False
    
    if not recommendation:
        print("❌ Failed to generate recommendation")
        return False
    
    print(f"\n✅ Generated recommendation ({len(recommendation)} characters)")
    print("\n" + "-" * 80)
    print("RECOMMENDATION PREVIEW:")
    print("-" * 80)
    print(recommendation[:500] + "..." if len(recommendation) > 500 else recommendation)
    print("-" * 80)
    
    # Save recommendation
    save_recommendation(recommendation, sentiment_analysis_dir, llm_method)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"📁 Output location: {os.path.join(sentiment_analysis_dir, 'recommendation')}")
    print("   Files created:")
    print("   - recommendation.txt (readable recommendation)")
    print("   - recommendation.json (structured data)")
    print("=" * 80 + "\n")
    
    return True
