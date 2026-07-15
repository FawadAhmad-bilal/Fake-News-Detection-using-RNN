# Fake News Detection using RNN

Classifies news articles as real or fake using a SimpleRNN-based sequence model trained on the Kaggle Fake and Real News Dataset.

## Dataset
- Kaggle: Fake and Real News Dataset (~44,000 articles)
- Combined `title` + `text` as input, labeled fake (0) / real (1)
- Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Approach
- Tokenized and padded article text (maxlen=250)
- Tokenizer fit only on training data to avoid data leakage
- Embedding layer (output_dim=2) + SimpleRNN(32) + Dense(sigmoid)
- Tested larger embedding sizes (32, 64) but they underperformed the smaller embedding on this dataset

## Results
- Validation Accuracy: 98.99%
- Confusion Matrix and Classification Report included in the notebook

## Note on Dataset Behavior
This dataset is known to contain strong stylistic shortcuts — most real articles come from Reuters and share consistent wire-service formatting, while fake articles don't. This likely explains why even a small embedding dimension performs very well: the model may be picking up on these surface-level patterns rather than deeper semantic understanding of "fake vs. real" content.

## Tech Stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## How to Run
1. Get a Kaggle API token from your Kaggle account settings
2. Set it as an environment variable or Colab secret (`KAGGLE_API_TOKEN`)
3. Run the notebook — it downloads the dataset, trains the model, and evaluates results
