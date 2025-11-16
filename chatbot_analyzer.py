#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Chatbot for Sentiment Analysis Results
Allows users to ask questions about their analysis using natural language

Created on Nov 6 2025
@author: andreyvlasenko
"""

import json
import os
from groq import Groq
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsChatbot:
    """
    Chatbot that answers questions about sentiment analysis results
    Uses RAG (Retrieval-Augmented Generation) over analysis data
    """
    
    def __init__(self, job_id: str, analysis_path: str, groq_api_key: str):
        """
        Initialize chatbot with analysis results
        
        Args:
            job_id: Unique job identifier
            analysis_path: Path to sentiment analysis results
            groq_api_key: Groq API key for LLM
        """
        self.job_id = job_id
        self.analysis_path = analysis_path
        self.groq_client = Groq(api_key=groq_api_key)
        self.context = self._load_analysis_context()
        self.conversation_history = []
        
        logger.info(f"Chatbot initialized for job {job_id}")
    
    def _load_analysis_context(self) -> Dict:
        """Load all analysis results as context for the chatbot"""
        context = {
            'job_id': self.job_id,
            'positive': {},
            'negative': {},
            'neutral': {},
            'recommendations': {},
            'trends': {},
            'statistics': {}
        }
        
        try:
            # Load positive sentiment data
            positive_path = os.path.join(self.analysis_path, 'positive')
            if os.path.exists(positive_path):
                context['positive'] = self._load_sentiment_data(positive_path, 'positive')
            
            # Load negative sentiment data
            negative_path = os.path.join(self.analysis_path, 'negative')
            if os.path.exists(negative_path):
                context['negative'] = self._load_sentiment_data(negative_path, 'negative')
            
            # Load neutral sentiment data
            neutral_path = os.path.join(self.analysis_path, 'neutral')
            if os.path.exists(neutral_path):
                context['neutral'] = self._load_sentiment_data(neutral_path, 'neutral')
            
            # Load recommendations
            rec_path = os.path.join(self.analysis_path, 'recommendation', 'recommendation.json')
            if os.path.exists(rec_path):
                with open(rec_path, 'r', encoding='utf-8') as f:
                    context['recommendations'] = json.load(f)
            
            # Load sentiment trends
            trends_path = os.path.join(self.analysis_path, 'sentiment_trends.json')
            if os.path.exists(trends_path):
                with open(trends_path, 'r', encoding='utf-8') as f:
                    context['trends'] = json.load(f)
            
            # Load representative comments
            rep_path = os.path.join(self.analysis_path, 'representative_comments.json')
            if os.path.exists(rep_path):
                with open(rep_path, 'r', encoding='utf-8') as f:
                    context['representatives'] = json.load(f)
            
            logger.info(f"Loaded context with {len(context)} data sources")
            
        except Exception as e:
            logger.error(f"Error loading analysis context: {e}")
        
        return context
    
    def _load_sentiment_data(self, path: str, sentiment_type: str) -> Dict:
        """Load data for a specific sentiment type"""
        data = {}
        
        # Load summary
        summary_file = os.path.join(path, f'{sentiment_type}_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                data['summary'] = json.load(f)
        
        # Load representatives
        rep_file = os.path.join(path, f'{sentiment_type}_representatives.json')
        if os.path.exists(rep_file):
            with open(rep_file, 'r', encoding='utf-8') as f:
                data['representatives'] = json.load(f)
        
        # Load top words
        words_file = os.path.join(path, f'{sentiment_type}_top_words.json')
        if os.path.exists(words_file):
            with open(words_file, 'r', encoding='utf-8') as f:
                data['top_words'] = json.load(f)
        
        return data
    
    def _build_context_prompt(self) -> str:
        """Build a concise context prompt from analysis data"""
        prompt_parts = []
        
        # Add statistics summary
        if self.context.get('trends'):
            trends = self.context['trends']
            summary = trends.get('summary', {})
            
            # Debug logging
            logger.info(f"Trends keys: {trends.keys()}")
            logger.info(f"Summary data: {summary}")
            
            total_positive = summary.get('total_positive', trends.get('positive_count', 0))
            total_negative = summary.get('total_negative', trends.get('negative_count', 0))
            total_neutral = summary.get('total_neutral', trends.get('neutral_count', 0))
            total_reviews = summary.get('total_reviews', total_positive + total_negative + total_neutral)
            
            logger.info(f"Counts - Positive: {total_positive}, Negative: {total_negative}, Neutral: {total_neutral}")
            
            # Calculate percentages
            if total_reviews > 0:
                pos_pct = (total_positive / total_reviews) * 100
                neg_pct = (total_negative / total_reviews) * 100
                neu_pct = (total_neutral / total_reviews) * 100
            else:
                pos_pct = neg_pct = neu_pct = 0
            
            prompt_parts.append(f"""
SENTIMENT DISTRIBUTION:
- Positive: {total_positive} reviews ({pos_pct:.1f}%)
- Negative: {total_negative} reviews ({neg_pct:.1f}%)
- Neutral: {total_neutral} reviews ({neu_pct:.1f}%)
- Total Reviews: {total_reviews}
""")
        
        # Add summaries for each sentiment
        for sentiment_type in ['positive', 'negative', 'neutral']:
            sentiment_data = self.context.get(sentiment_type, {})
            if sentiment_data.get('summary'):
                summary_text = sentiment_data['summary'].get('summary', '')
                if summary_text:
                    prompt_parts.append(f"\n{sentiment_type.upper()} FEEDBACK SUMMARY:\n{summary_text}")
            
            # Add top keywords
            if sentiment_data.get('top_words'):
                words = sentiment_data['top_words'][:10]  # Top 10 words
                words_str = ', '.join([f"{w['word']} ({w['count']})" for w in words])
                prompt_parts.append(f"{sentiment_type.upper()} Keywords: {words_str}")
            
            # Add representative examples (limit to 3 per sentiment)
            if sentiment_data.get('representatives'):
                reps = sentiment_data['representatives'][:3]
                examples = [f"- \"{r.get('text', '')}\"" for r in reps]
                prompt_parts.append(f"\n{sentiment_type.upper()} Examples:\n" + "\n".join(examples))
        
        # Add recommendations
        if self.context.get('recommendations'):
            rec = self.context['recommendations'].get('recommendations', '')
            if rec:
                prompt_parts.append(f"\nRECOMMENDATIONS:\n{rec}")
        
        return "\n".join(prompt_parts)
    
    def ask(self, question: str, include_history: bool = True) -> str:
        """
        Ask a question about the analysis results
        
        Args:
            question: User's question in natural language
            include_history: Whether to include conversation history
            
        Returns:
            AI-generated answer based on analysis data
        """
        try:
            # Build system prompt with context
            context_prompt = self._build_context_prompt()
            
            system_message = f"""You are an expert sentiment analysis assistant. You help users understand their customer feedback data.

You have access to the following sentiment analysis results:

{context_prompt}

Guidelines:
- Provide clear, actionable insights
- Use specific data and examples from the analysis
- Be concise but thorough
- If data is missing, say so
- Suggest follow-up questions when appropriate
- Focus on actionable recommendations

Answer questions based ONLY on the data provided above."""

            # Build messages list
            messages = [{"role": "system", "content": system_message}]
            
            # Add conversation history if requested
            if include_history and self.conversation_history:
                messages.extend(self.conversation_history[-4:])  # Last 2 exchanges
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            # Get response from Groq
            logger.info(f"Sending question to Groq: {question}")
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fast and reliable (recommended for gemma2-9b-it replacement)
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1024,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            logger.info(f"Generated answer of length {len(answer)}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}"
    
    def get_suggested_questions(self) -> List[str]:
        """Get a list of suggested questions users can ask"""
        suggestions = [
            "What are the main issues customers are complaining about?",
            "What do customers like most about the service?",
            "What should we prioritize fixing first?",
            "Show me examples of negative feedback about food quality",
            "What percentage of reviews are positive?",
            "What are the common themes in negative reviews?",
            "What improvements would have the biggest impact?",
            "Are there any seasonal trends in the sentiment?",
            "What specific words appear most in negative reviews?",
            "How does the positive feedback compare to negative?"
        ]
        
        # Customize based on available data
        if self.context.get('negative', {}).get('summary'):
            suggestions.insert(0, "Summarize the negative feedback")
        
        if self.context.get('recommendations'):
            suggestions.insert(1, "What are your top recommendations?")
        
        return suggestions[:8]  # Return top 8
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_context_summary(self) -> Dict:
        """Get a summary of available context data"""
        return {
            'job_id': self.job_id,
            'has_positive': bool(self.context.get('positive')),
            'has_negative': bool(self.context.get('negative')),
            'has_neutral': bool(self.context.get('neutral')),
            'has_recommendations': bool(self.context.get('recommendations')),
            'has_trends': bool(self.context.get('trends')),
            'conversation_length': len(self.conversation_history)
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python chatbot_analyzer.py <job_id> <analysis_path>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    analysis_path = sys.argv[2]
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize chatbot
    chatbot = ResultsChatbot(job_id, analysis_path, groq_api_key)
    
    print("=" * 80)
    print("SENTIMENT ANALYSIS CHATBOT")
    print("=" * 80)
    print(f"\nAnalyzing results for job: {job_id}")
    print("\nSuggested questions:")
    for i, question in enumerate(chatbot.get_suggested_questions(), 1):
        print(f"{i}. {question}")
    
    print("\nType 'quit' to exit, 'clear' to clear history\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Conversation history cleared.")
                continue
            
            if not user_input:
                continue
            
            # Get answer
            answer = chatbot.ask(user_input)
            print(f"\nAssistant: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
