#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re





def vizualization(sentiment_counts, 
                  results_df,
                  processing_times,
                  folders,
                  WORDCLOUD_MAX_WORDS, 
                  TOP_WORDS_COUNT, 
                  df_sample,
                  total_time,
                  OUTPUT_BASE_DIR,
                  representative_results,
                  trends=None):
    
    # Print trends variable content for debugging
    print("\n" + "=" * 80)
    print("TRENDS VARIABLE CONTENT")
    print("=" * 80)
    if trends:
        print(f"Number of dates in trends: {len(trends)}")
        print(f"\nFirst 10 trend entries:")
        for i, trend in enumerate(trends[:10], 1):
            print(f"  {i}. Date: {trend['date']} | Positive: {trend['positive']} | Negative: {trend['negative']} | Neutral: {trend['neutral']} | Total: {trend['total']}")
        if len(trends) > 10:
            print(f"  ... and {len(trends) - 10} more dates")
    else:
        print("⚠️  No trends data available")
    print("=" * 80)
    
    # Create Visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Sentiment Distribution Pie Chart
    print("\n📊 Creating sentiment distribution chart...")

    plt.figure(figsize=(12, 8))

    # Pie chart
    plt.subplot(2, 2, 1)
    sizes = [sentiment_counts[s] for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']]
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05)

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=explode, shadow=True)
    plt.title('Sentiment Distribution (Filtered Reviews)', fontsize=14, fontweight='bold')

    # 2. Confidence Distribution
    plt.subplot(2, 2, 2)
    confidences = results_df['confidence'].values
    plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(confidences.mean(), color='red', linestyle='--', 
               label=f'Mean: {confidences.mean():.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()

    # 3. Sentiment vs Original Quality Score (instead of accuracy visualization)
    plt.subplot(2, 2, 3)
    if 'original_score' in results_df.columns:
        # Create scatter plot of sentiment vs original quality score
        sentiment_colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_data = results_df[results_df['sentiment'] == sentiment]
            if len(sentiment_data) > 0:
                plt.scatter(sentiment_data['original_score'], sentiment_data['confidence'], 
                           c=sentiment_colors[sentiment], label=sentiment, alpha=0.6)
        plt.xlabel('Original Quality Score')
        plt.ylabel('Sentiment Confidence')
        plt.title('Sentiment vs Original Quality Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Fallback to simple sentiment distribution
        sentiment_counts_plot = results_df['sentiment'].value_counts()
        plt.bar(sentiment_counts_plot.index, sentiment_counts_plot.values, 
                color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.title('Sentiment Distribution')
        plt.ylabel('Count')

    # 4. Sentiment Trends Over Time (replaces Processing Time Analysis)
    plt.subplot(2, 2, 4)
    if trends and len(trends) > 0:
        # Extract data from trends
        dates = [t['date'] for t in trends]
        positive_counts = [t['positive'] for t in trends]
        negative_counts = [t['negative'] for t in trends]
        neutral_counts = [t['neutral'] for t in trends]
        
        # Convert dates to display format (show only day numbers for clarity)
        date_labels = [date.split('-')[-1] for date in dates]  # Extract day number
        
        # Plot lines for each sentiment
        x_positions = range(len(dates))
        plt.plot(x_positions, positive_counts, marker='o', linewidth=2, markersize=6, 
                color='#2ecc71', label='Positive', alpha=0.8)
        plt.plot(x_positions, negative_counts, marker='s', linewidth=2, markersize=6, 
                color='#e74c3c', label='Negative', alpha=0.8)
        plt.plot(x_positions, neutral_counts, marker='^', linewidth=2, markersize=6, 
                color='#95a5a6', label='Neutral', alpha=0.8)
        
        plt.xlabel('Day of Month')
        plt.ylabel('Number of Reviews')
        plt.title('Sentiment Trends Over Time')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks to show dates (every nth date to avoid crowding)
        step = max(1, len(dates) // 10)  # Show ~10 labels maximum
        plt.xticks([i for i in range(0, len(dates), step)], 
                   [date_labels[i] for i in range(0, len(dates), step)],
                   rotation=45)
        
        print("✅ Trends plot created successfully")
    else:
        # Fallback message if no trends
        plt.text(0.5, 0.5, 'No trends data available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, color='gray')
        plt.title('Sentiment Trends Over Time')
        print("⚠️  No trends data - showing placeholder")

    plt.tight_layout()
    plt.savefig(os.path.join(folders['visualizations'], 'sentiment_analysis_overview.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

    # Additional detailed trends visualization if data is available
    if trends and len(trends) > 0:
        print("\n📈 Creating detailed sentiment trends visualization...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Extract data
        dates = [t['date'] for t in trends]
        positive_counts = [t['positive'] for t in trends]
        negative_counts = [t['negative'] for t in trends]
        neutral_counts = [t['neutral'] for t in trends]
        total_counts = [t['total'] for t in trends]
        
        date_labels = [date.split('-')[-1] for date in dates]
        x_positions = range(len(dates))
        
        # Plot 1: Stacked area chart
        ax1 = axes[0]
        ax1.fill_between(x_positions, 0, positive_counts, color='#2ecc71', alpha=0.7, label='Positive')
        ax1.fill_between(x_positions, positive_counts, 
                        [p + n for p, n in zip(positive_counts, neutral_counts)], 
                        color='#95a5a6', alpha=0.7, label='Neutral')
        ax1.fill_between(x_positions, [p + n for p, n in zip(positive_counts, neutral_counts)],
                        [p + n + neg for p, n, neg in zip(positive_counts, neutral_counts, negative_counts)],
                        color='#e74c3c', alpha=0.7, label='Negative')
        
        ax1.set_xlabel('Day of Month', fontsize=11)
        ax1.set_ylabel('Number of Reviews', fontsize=11)
        ax1.set_title('Sentiment Distribution Over Time (Stacked)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        step = max(1, len(dates) // 15)
        ax1.set_xticks([i for i in range(0, len(dates), step)])
        ax1.set_xticklabels([date_labels[i] for i in range(0, len(dates), step)], rotation=45)
        
        # Plot 2: Sentiment ratio (positive - negative) / total
        ax2 = axes[1]
        sentiment_ratio = [(p - n) / t if t > 0 else 0 
                          for p, n, t in zip(positive_counts, negative_counts, total_counts)]
        
        colors = ['#2ecc71' if ratio > 0 else '#e74c3c' for ratio in sentiment_ratio]
        ax2.bar(x_positions, sentiment_ratio, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_xlabel('Day of Month', fontsize=11)
        ax2.set_ylabel('Sentiment Ratio', fontsize=11)
        ax2.set_title('Daily Sentiment Score ((Positive - Negative) / Total)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax2.set_xticks([i for i in range(0, len(dates), step)])
        ax2.set_xticklabels([date_labels[i] for i in range(0, len(dates), step)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(folders['visualizations'], 'sentiment_trends_detailed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Detailed trends visualization saved")

    # 2. Word Clouds for each sentiment
    print("\n☁️  Creating word clouds...")

    def clean_text_for_wordcloud(text):
        """Clean text for word cloud generation"""
        # Remove URLs, mentions, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, sentiment in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL']):
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        
        if len(sentiment_data) > 0:
            # Combine all texts
            all_text = ' '.join([clean_text_for_wordcloud(text) for text in sentiment_data['text']])
            
            if all_text.strip():  # Check if we have text after cleaning
                wordcloud = WordCloud(
                    width=400, height=300, 
                    background_color='white',
                    colormap=['Greens', 'Reds', 'Greys'][i],
                    max_words=WORDCLOUD_MAX_WORDS,
                    relative_scaling=0.5,
                    random_state=42
                ).generate(all_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment} Words', fontsize=14, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment}\ndata available', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{sentiment} Words', fontsize=14)
                axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folders['visualizations'], 'sentiment_wordclouds.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Most common words analysis
    print("\n📈 Creating word frequency analysis...")

    def get_top_words(texts, n_words=None):
        """Get top words from texts"""
        if n_words is None:
            n_words = TOP_WORDS_COUNT
        all_text = ' '.join([clean_text_for_wordcloud(text) for text in texts])
        words = all_text.split()
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs'}
        words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        return Counter(words).most_common(n_words)

    plt.figure(figsize=(15, 10))

    for i, sentiment in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 1):
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        
        if len(sentiment_data) > 0:
            top_words = get_top_words(sentiment_data['text'])
            
            if top_words:
                words, counts = zip(*top_words)
                
                plt.subplot(2, 2, i)
                bars = plt.bar(range(len(words)), counts, 
                              color=['#2ecc71', '#e74c3c', '#95a5a6'][i-1], alpha=0.7)
                plt.xlabel('Words')
                plt.ylabel('Frequency')
                plt.title(f'Top Words in {sentiment} Comments')
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom', fontsize=9)

    # 4. Confidence distribution by sentiment
    plt.subplot(2, 2, 4)
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        if len(sentiment_data) > 0:
            plt.hist(sentiment_data['confidence'], alpha=0.6, label=sentiment, bins=20)

    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Sentiment')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folders['visualizations'], 'word_frequency_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

    # Save comprehensive results
    print("\n💾 Saving comprehensive results...")

    # Performance summary
    performance_summary = {
        'total_samples': len(df_sample),
        'processing_time_minutes': total_time / 60,
        'avg_time_per_sample': total_time / len(df_sample),
        'sentiment_distribution': sentiment_counts,
        'score_distribution': {
            'avg_original_score': float(results_df['original_score'].mean()) if 'original_score' in results_df.columns else 0,
            'avg_sentiment_confidence': float(results_df['confidence'].mean()),
            'candidates_count': int(results_df['is_candidate'].sum()) if 'is_candidate' in results_df.columns else 0
        },
        'confidence_stats': {
            'mean': float(results_df['confidence'].mean()),
            'std': float(results_df['confidence'].std()),
            'min': float(results_df['confidence'].min()),
            'max': float(results_df['confidence'].max())
        }
    }

    # Save performance summary
    with open(os.path.join(OUTPUT_BASE_DIR, 'performance_summary.json'), 'w') as f:
        json.dump(performance_summary, f, indent=2)

    # Save full results
    results_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'complete_results.csv'), index=False)

    # Save representative comments summary
    representatives_summary = {}
    for sentiment, representatives in representative_results.items():
        if len(representatives) > 0:
            representatives_summary[sentiment] = representatives[['text', 'confidence', 'cluster_id', 'cluster_size']].to_dict('records')

    with open(os.path.join(OUTPUT_BASE_DIR, 'representative_comments.json'), 'w', encoding='utf-8') as f:
        json.dump(representatives_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All results saved to: {OUTPUT_BASE_DIR}")

    print("\n" + "=" * 80)
    print("SUMMARY OF REPRESENTATIVE COMMENTS")
    print("=" * 80)

    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment in representative_results and len(representative_results[sentiment]) > 0:
            print(f"\n🎯 MOST REPRESENTATIVE {sentiment} COMMENTS:")
            print("-" * 50)
            
            representatives = representative_results[sentiment].sort_values('confidence', ascending=False)
            
            for i, (_, row) in enumerate(representatives.head(5).iterrows(), 1):
                print(f"\n{i}. [Confidence: {row['confidence']:.3f} | Cluster: {row['cluster_id']} | Size: {row['cluster_size']}]")
                print(f"   \"{row['text']}\"")

    print(f"\n" + "=" * 80)
    print("✅ ENHANCED ANALYSIS COMPLETE!")
    print("=" * 80)

    print(f"""
    📊 Analysis Summary:
       • Processed: {len(df_sample):,} samples
       • Processing time: {total_time/60:.1f} minutes
       • Average original quality score: {performance_summary['score_distribution']['avg_original_score']:.3f}
       • Average sentiment confidence: {performance_summary['score_distribution']['avg_sentiment_confidence']:.3f}
       • Files created: {len(os.listdir(OUTPUT_BASE_DIR))} output files
       
    📁 Output Structure:
       • {OUTPUT_BASE_DIR}/positive/ - Positive sentiment data
       • {OUTPUT_BASE_DIR}/negative/ - Negative sentiment data  
       • {OUTPUT_BASE_DIR}/neutral/ - Neutral sentiment data
       • {OUTPUT_BASE_DIR}/visualizations/ - Charts and graphs
       • {OUTPUT_BASE_DIR}/vectors/ - Vector analysis data
       
    🎯 Key Insights:
       • Most confident positive: {results_df[results_df['sentiment']=='POSITIVE']['confidence'].max():.3f}
       • Most confident negative: {results_df[results_df['sentiment']=='NEGATIVE']['confidence'].max():.3f}
       • Neutral classifications: {sentiment_counts['NEUTRAL']} ({sentiment_counts['NEUTRAL']/len(results_df)*100:.1f}%)
       • High-quality candidates: {performance_summary['score_distribution']['candidates_count']}
    """)

    print(f"\n� All outputs saved to: {OUTPUT_BASE_DIR}")
    print("🎉 Complete analysis package ready for review!")
    print("📄 Note: Use the separate generate_pdf_only.py script to create PDF reports")