# Mining Consumer Voices: A Sentiment Analysis of Financial Complaints Using NLP

This project analyzes consumer complaint data from the Consumer Financial Protection Bureau (CFPB) using Natural Language Processing (NLP). The goal is to extract insights from complaint narratives, classify sentiment, and identify trends in consumer satisfaction across financial products and companies.

---

## 📌 Project Objectives

- Analyze and preprocess real-world consumer complaint data.
- Perform sentiment analysis on complaint narratives based on company responses.
- Extract key themes and concerns for different financial products using TF-IDF.
- Generate word clouds and summaries for each product category.
- Identify the best and worst-performing companies based on sentiment trends.

---

## 🗂 Dataset

**Source:** [CFPB Consumer Complaints](https://www.kaggle.com/datasets/sbhatti/consumer-finance-complaints)

**Key Features Used:**

- `consumer_complaint_narrative`
- `product`
- `company`
- `company_response_to_consumer`
- `submitted_via`
- `timely_response`
- `consumer_disputed?`

---

## 🛠️ Methods & Tools

| Task                        | Libraries/Techniques                            |
|----------------------------|--------------------------------------------------|
| Data cleaning              | Pandas, NumPy                                    |
| Text preprocessing         | NLTK, Regex (`re`), string                       |
| Sentiment classification   | Rule-based mapping from company responses        |
| Keyword extraction         | TF-IDF (scikit-learn)                            |
| Visualization              | Matplotlib, WordCloud                            |
| Text summarization         | Sumy / Transformers (optional)                   |
| Development environment    | Google Colab                                     |

---

## 🧹 Preprocessing Steps

1. Drop rows with missing narratives for NLP.
2. Normalize text:
   - Lowercasing
   - Remove URLs, digits, punctuation
   - Tokenization
   - Stopword removal
   - Lemmatization
3. Clean and prepare separate datasets for:
   - **NLP-based Sentiment Analysis**
   - **Exploratory Data Analysis (EDA)**

---

## 🧠 Sentiment Analysis Logic

Sentiment is inferred from the company’s response to the complaint:

```python
def map_sentiment(response):
    if 'monetary relief' in response.lower():
        return 'Positive'
    elif 'no relief' in response.lower():
        return 'Negative'
    elif 'explanation' in response.lower():
        return 'Neutral'
    else:
        return 'Neutral'
