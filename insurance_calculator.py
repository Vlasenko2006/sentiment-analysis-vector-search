"""
Insurance Risk Assessment Module

Calculates bankruptcy insurance cost for restaurants based on sentiment analysis data.
Uses sentiment distribution, confidence statistics, and temporal trends to assess risk.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def calculate_insurance_risk(
    performance_summary: Dict,
    sentiment_trends: Dict,
    base_rate: float = 5000.0  # Base insurance cost in USD
) -> Dict:
    """
    Calculate bankruptcy insurance risk and cost based on sentiment analysis.
    
    Args:
        performance_summary: Dict with sentiment_distribution, confidence_stats, total_samples
        sentiment_trends: Dict with trends (daily sentiment data) and summary
        base_rate: Base insurance cost before risk multipliers (default: $5000)
    
    Returns:
        Dict containing:
            - insurance_cost: Final calculated insurance cost
            - risk_level: Low/Medium/High/Critical
            - risk_score: Numerical risk score (0-100)
            - breakdown: Dict with detailed calculation factors
    """
    
    # Extract sentiment distribution
    sentiment_dist = performance_summary.get('sentiment_distribution', {})
    total_samples = performance_summary.get('total_samples', 0)
    confidence_stats = performance_summary.get('confidence_stats', {})
    
    # Calculate sentiment percentages
    positive_count = sentiment_dist.get('POSITIVE', 0)
    negative_count = sentiment_dist.get('NEGATIVE', 0)
    neutral_count = sentiment_dist.get('NEUTRAL', 0)
    
    if total_samples == 0:
        return {
            'insurance_cost': base_rate,
            'risk_level': 'Unknown',
            'risk_score': 50,
            'breakdown': {'error': 'No samples available'}
        }
    
    positive_ratio = positive_count / total_samples
    negative_ratio = negative_count / total_samples
    neutral_ratio = neutral_count / total_samples
    
    # Confidence factor (lower confidence = higher risk)
    avg_confidence = confidence_stats.get('mean', 0.9)
    confidence_std = confidence_stats.get('std', 0.1)
    min_confidence = confidence_stats.get('min', 0.5)
    
    # Confidence multiplier: ranges from 1.0 (high confidence) to 1.5 (low confidence)
    confidence_multiplier = 1.5 - (avg_confidence * 0.5)
    
    # Low confidence variance adds uncertainty risk
    if confidence_std > 0.2:
        confidence_multiplier *= 1.1
    
    # Sentiment risk factor
    # High negative ratio increases risk significantly
    # Neutral treated as slightly negative (ambivalence)
    sentiment_multiplier = 1.0 + (negative_ratio * 2.5) + (neutral_ratio * 0.5)
    
    # Very high positive ratio reduces risk
    if positive_ratio > 0.85:
        sentiment_multiplier *= 0.85
    elif positive_ratio > 0.75:
        sentiment_multiplier *= 0.95
    
    # Sample size factor (small sample = higher uncertainty)
    if total_samples < 50:
        sample_multiplier = 1.3
    elif total_samples < 100:
        sample_multiplier = 1.15
    else:
        sample_multiplier = 1.0
    
    # Trend analysis - check for recent negative patterns
    trend_multiplier = _analyze_trend_risk(sentiment_trends)
    
    # Calculate final insurance cost
    insurance_cost = base_rate * sentiment_multiplier * confidence_multiplier * sample_multiplier * trend_multiplier
    
    # Calculate risk score (0-100)
    risk_score = _calculate_risk_score(
        positive_ratio, negative_ratio, avg_confidence, 
        total_samples, trend_multiplier
    )
    
    # Determine risk level
    risk_level = _determine_risk_level(risk_score)
    
    # Detailed breakdown
    breakdown = {
        'base_rate': base_rate,
        'sentiment_factors': {
            'positive_percentage': round(positive_ratio * 100, 1),
            'negative_percentage': round(negative_ratio * 100, 1),
            'neutral_percentage': round(neutral_ratio * 100, 1),
            'sentiment_multiplier': round(sentiment_multiplier, 2)
        },
        'confidence_factors': {
            'average_confidence': round(avg_confidence, 3),
            'confidence_std': round(confidence_std, 3),
            'min_confidence': round(min_confidence, 3),
            'confidence_multiplier': round(confidence_multiplier, 2)
        },
        'sample_factors': {
            'total_samples': total_samples,
            'sample_multiplier': round(sample_multiplier, 2)
        },
        'trend_factors': {
            'trend_multiplier': round(trend_multiplier, 2),
            'trend_status': _get_trend_status(trend_multiplier)
        }
    }
    
    return {
        'insurance_cost': round(insurance_cost, 2),
        'risk_level': risk_level,
        'risk_score': risk_score,
        'breakdown': breakdown
    }


def _analyze_trend_risk(sentiment_trends: Dict) -> float:
    """
    Analyze temporal trends to detect recent negative patterns.
    
    Returns:
        Trend multiplier: 0.9 (improving), 1.0 (stable), 1.2-1.4 (deteriorating)
    """
    trends = sentiment_trends.get('trends', [])
    if not trends or len(trends) < 7:
        return 1.0  # Insufficient data for trend analysis
    
    # Analyze last 14 days vs previous period
    recent_days = trends[-14:] if len(trends) >= 14 else trends
    previous_days = trends[-28:-14] if len(trends) >= 28 else []
    
    # Calculate negative ratio in recent period
    recent_total = sum(day['total'] for day in recent_days)
    recent_negative = sum(day['negative'] for day in recent_days)
    
    if recent_total == 0:
        return 1.0
    
    recent_negative_ratio = recent_negative / recent_total
    
    # Compare to previous period if available
    if previous_days:
        prev_total = sum(day['total'] for day in previous_days)
        prev_negative = sum(day['negative'] for day in previous_days)
        
        if prev_total > 0:
            prev_negative_ratio = prev_negative / prev_total
            
            # Deteriorating trend
            if recent_negative_ratio > prev_negative_ratio * 1.5:
                return 1.4  # Significant increase in negative reviews
            elif recent_negative_ratio > prev_negative_ratio * 1.2:
                return 1.2  # Moderate increase
            # Improving trend
            elif recent_negative_ratio < prev_negative_ratio * 0.7:
                return 0.9  # Significant improvement
    
    # Check for recent spike in negatives (last 3 days)
    last_3_days = trends[-3:]
    last_3_total = sum(day['total'] for day in last_3_days)
    last_3_negative = sum(day['negative'] for day in last_3_days)
    
    if last_3_total > 0:
        last_3_negative_ratio = last_3_negative / last_3_total
        if last_3_negative_ratio > 0.3:  # More than 30% negative in last 3 days
            return 1.3
    
    return 1.0  # Stable trend


def _calculate_risk_score(
    positive_ratio: float, 
    negative_ratio: float, 
    confidence: float,
    sample_size: int,
    trend_multiplier: float
) -> int:
    """
    Calculate numerical risk score from 0 (lowest risk) to 100 (highest risk).
    """
    # Start with base score from negative ratio (0-40 points)
    score = negative_ratio * 200  # 20% negative = 40 points
    
    # Add points for low positive ratio (0-20 points)
    if positive_ratio < 0.6:
        score += (0.6 - positive_ratio) * 50
    
    # Add points for low confidence (0-20 points)
    if confidence < 0.9:
        score += (0.9 - confidence) * 100
    
    # Add points for small sample size (0-10 points)
    if sample_size < 100:
        score += (100 - sample_size) / 10
    
    # Add points for negative trend (0-10 points)
    if trend_multiplier > 1.0:
        score += (trend_multiplier - 1.0) * 25
    
    # Cap at 100
    return min(int(score), 100)


def _determine_risk_level(risk_score: int) -> str:
    """Determine risk level based on score."""
    if risk_score >= 70:
        return 'Critical'
    elif risk_score >= 50:
        return 'High'
    elif risk_score >= 30:
        return 'Medium'
    else:
        return 'Low'


def _get_trend_status(trend_multiplier: float) -> str:
    """Get human-readable trend status."""
    if trend_multiplier >= 1.3:
        return 'Significantly deteriorating'
    elif trend_multiplier >= 1.1:
        return 'Moderately deteriorating'
    elif trend_multiplier <= 0.95:
        return 'Improving'
    else:
        return 'Stable'


# Test function
if __name__ == "__main__":
    # Test with sample data
    test_performance = {
        "total_samples": 131,
        "sentiment_distribution": {"POSITIVE": 106, "NEGATIVE": 23, "NEUTRAL": 2},
        "confidence_stats": {"mean": 0.989, "std": 0.049, "min": 0.583, "max": 0.999}
    }
    
    test_trends = {
        "trends": [
            {"date": "2025-10-01", "positive": 2, "negative": 0, "neutral": 0, "total": 2},
            {"date": "2025-10-02", "positive": 1, "negative": 0, "neutral": 0, "total": 1}
        ],
        "summary": {
            "total_dates": 28,
            "date_range": {"start": "2025-10-01", "end": "2025-10-31"},
            "total_reviews": 36,
            "total_positive": 33,
            "total_negative": 3,
            "total_neutral": 0
        }
    }
    
    result = calculate_insurance_risk(test_performance, test_trends, base_rate=5000.0)
    
    print("=== Insurance Risk Assessment ===")
    print(f"Insurance Cost: ${result['insurance_cost']:,.2f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']}/100")
    print("\nBreakdown:")
    print(json.dumps(result['breakdown'], indent=2))
