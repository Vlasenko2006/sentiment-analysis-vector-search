# Enhanced Sentiment Analysis with Vector Search

A comprehensive sentiment analysis project using DistilBERT and advanced vector search techniques to analyze Twitter data from the Sentiment140 dataset.

## 🚀 Features

- **Multi-class Sentiment Analysis**: POSITIVE, NEGATIVE, and NEUTRAL classification
- **Vector Search**: Find most representative comments using TF-IDF and K-means clustering
- **Large Scale Processing**: Analyze 3,500+ tweets efficiently
- **Rich Visualizations**: Word clouds, distribution charts, and frequency analysis
- **Organized Output**: Automatic indexing of comments by sentiment
- **Performance Metrics**: Comprehensive accuracy and confidence analysis

## 📊 Results Overview

- **Processing Speed**: ~41 tweets per second
- **Dataset**: Sentiment140 (1.6M Twitter messages)
- **Sample Size**: 3,500 balanced samples
- **Accuracy**: 67.7% overall accuracy
- **Distribution**: 58.4% negative, 35.7% positive, 5.9% neutral

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- conda or pip for package management

### Setup Environment

```bash
# Create conda environment
conda create -n sentiment_analysis python=3.10
conda activate sentiment_analysis

# Install required packages
pip install transformers pandas numpy scikit-learn matplotlib seaborn wordcloud
```

### Download the Dataset

The scripts will automatically download the Sentiment140 dataset from Stanford University.

## 📁 Project Structure

```
sentiment-analysis/
├── README.md
├── requirements.txt
├── download_and_prepare_dataset.py     # Dataset download utility
├── request_simple.py                   # Basic sentiment analysis
├── request_RoBERTa.py                 # 3-class sentiment analysis
├── request_RoBERTa_enhanced.py        # Enhanced version with vector search
├── test_capabilities.py               # Model capability testing
├── reports/                           # Analysis reports
│   ├── sentiment_analysis_capabilities_report.md
│   └── roberta_validation_report.md
└── my_volume/                         # Output directory
    ├── sentiment140.csv               # Downloaded dataset
    ├── hf_model/                      # Cached model files
    └── sentiment_analysis/            # Analysis results
        ├── positive/                  # Positive sentiment data
        ├── negative/                  # Negative sentiment data
        ├── neutral/                   # Neutral sentiment data
        └── visualizations/            # Generated charts
```

## 🚀 Usage

### Basic Sentiment Analysis

```bash
python request_simple.py
```

Performs basic sentiment analysis on sample texts and tests the model capabilities.

### Enhanced Analysis with Vector Search

```bash
python request_RoBERTa_enhanced.py
```

Processes 3,500 tweets, creates sentiment indexes, finds representative comments, and generates visualizations.

### Test Model Capabilities

```bash
python test_capabilities.py
```

Tests the model's ability to handle emojis, punctuation, internet slang, and various text types.

## 📈 Key Features

### 1. Representative Comment Discovery

Using vector search with TF-IDF and K-means clustering to find the most representative comments:

**Most Representative Positive:**
- "i love my grandma" (Confidence: 1.000)
- "herbal essence smells so good" (Confidence: 1.000)

**Most Representative Negative:**
- "im sad now Miss.Lilly" (Confidence: 0.996)
- "It was a sleepless night" (Confidence: 0.996)

### 2. Advanced 3-Class Simulation

Converts binary sentiment model to 3-class using confidence thresholds:
- High confidence (>0.8) → Original prediction
- Low confidence (≤0.8) → NEUTRAL

### 3. Comprehensive Visualizations

- Sentiment distribution pie charts
- Word clouds for each sentiment
- Confidence score histograms
- Word frequency analysis
- Processing performance metrics

## 🎯 Model Performance

### Overall Metrics
- **Accuracy**: 67.7% on balanced dataset
- **Processing Speed**: 1.4 minutes for 3,500 samples
- **Confidence**: 97.3% average confidence

### Sentiment Distribution
- **Negative**: 2,043 comments (58.4%)
- **Positive**: 1,249 comments (35.7%)
- **Neutral**: 208 comments (5.9%)

## 🔬 Technical Details

### Model Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Framework**: HuggingFace Transformers
- **Training Data**: Stanford Sentiment Treebank v2
- **Model Size**: ~250MB

### Vector Search Implementation
- **Vectorization**: TF-IDF with 1000 features
- **Clustering**: K-means with 10 clusters per sentiment
- **Similarity**: Cosine similarity to find centroids
- **Representative Selection**: Closest to cluster centroid

## 📊 Capabilities Analysis

The model successfully handles:
- ✅ Plain text sentiment
- ✅ Emojis and emoticons (😊, 😞, 👍, 👎)
- ✅ Internet slang ("LOL", "OMG", "FML")
- ✅ Punctuation patterns (!!!, ...)
- ✅ Star ratings and scores
- ✅ Mixed language content

## 📝 Reports

Detailed analysis reports are available in the `reports/` directory:

1. **Model Capabilities Report**: Comprehensive testing of what the model can analyze
2. **Validation Report**: Performance metrics on real Twitter data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stanford University for the Sentiment140 dataset
- HuggingFace for the transformers library
- The open-source community for the various tools and libraries used

Project Link: [https://github.com/yourusername/sentiment-analysis-vector-search](https://github.com/yourusername/sentiment-analysis-vector-search)

---

**Note**: This project is for educational and research purposes. The Sentiment140 dataset is used under Stanford University's terms of use.
