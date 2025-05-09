# Judge a Book by its cover   
*Srishti Murarka*  
DA623: Multimodal Data Analysis (Winter 2025)  
Date: May 9, 2025

---

## Motivation

With the rapid growth of digital libraries and online bookstores, automatic genre classification of books has become increasingly important for search, recommendation, and organization. I chose this project because it asks a deceptively simple question: **Can we predict a book’s genre using only its title?** Titles are often short, creative, and ambiguous, making this a challenging and interesting problem for machine learning and natural language processing. This project also connects to real-world applications in information retrieval and recommender systems, and provides a foundation for exploring multimodal learning.

---

## Historical Perspective

Book genre classification has a rich history across several modalities:

- **Text-based approaches:** Early work used book summaries or full text for genre prediction, achieving high accuracy but requiring extensive data and computation.
- **Image-based approaches:** Recent projects use book cover images with deep learning (e.g., VGG16) for genre prediction, as in [HimanshuRaj98/book-genre-classification][3].
- **Multimodal approaches:** Combining titles, summaries, and cover images has shown improved accuracy, leveraging the strengths of each data type.
- **Title-only approaches:** Some studies, including this project, focus on titles for their accessibility and speed, though with lower accuracy compared to richer inputs.

This project benchmarks classic and deep learning models using only book titles, and discusses how multimodal data could further improve results.

---

## Learning & Explanation

### Dataset

- **Source:** [Judging a Book by its Cover][2]
- **Size:** 207,572 books, 32 genres
- **Fields:** ASIN, image URL, title, author, genre
- **For this project:** Only the title is used as input, with the genre as the label.

Example data row:

| Label | Category Name                | Size   |
|-------|-----------------------------|--------|
| 4     | Children's Books            | 13,605 |
| 22    | Romance                     | 4,291  |
| 23    | Science & Math              | 9,276  |
| 17    | Mystery, Thriller & Suspense| 1,998  |
| 29    | Travel                      | 18,338 |
| ...   | ...                         | ...    |

### Algorithms Used

1. **Bag-of-Words + Feed-Forward Neural Network**  
   - Converts titles into word count vectors (ignores order/context).
   - Simple neural network for classification.

2. **TF-IDF + Classical ML Models**  
   - TF-IDF weighs rare words more heavily.
   - Models: Logistic Regression, Multinomial Naive Bayes, Multi-Layer Perceptron, XGBoost.

3. **RNN/LSTM + GloVe Embeddings**  
   - Uses pre-trained GloVe vectors for word embeddings.
   - RNNs/LSTMs capture word order and context in titles.

---

## Code / Experiments

### Data Loading & Preprocessing
```import pandas as pd```
### Load the dataset
```df = pd.read_csv('books.csv', header=None, names=['ASIN', 'FILENAME', 'IMAGE_URL', 'TITLE', 'AUTHOR', 'CATEGORY_ID', 'CATEGORY'])```
### Only keep title and genre
```df = df[['TITLE', 'CATEGORY']]```
```df = df.dropna()```
### Show class distribution
```df['CATEGORY'].value_counts().plot(kind='bar', figsize=(12,4), title='Genre Distribution')```

### TF-IDF + Logistic Regression

```from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df['CLEAN_TITLE']
y = df['CATEGORY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```


### Model Comparison (Summary Table)

| Model                           | Accuracy   |
|----------------------------------|------------|
| Bag-of-Words + NN               | ~60%       |
| TF-IDF + Logistic Regression    | ~65%       |
| TF-IDF + Naive Bayes            | ~62%       |
| RNN/LSTM + GloVe                | ~66%       |

*Note: Actual results may vary depending on data splits and preprocessing.*

### Visualization: Confusion Matrix

```import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(14,10))
sns.heatmap(cm, xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```


---

## Reflections

### What Surprised Me

- **Short titles are limiting:** Many titles are too vague or creative to give strong genre cues.
- **Classical models are strong:** TF-IDF with Logistic Regression is a surprisingly strong baseline.
- **Deep learning is not always better:** With only titles, deep models like LSTM do not outperform classical models by much.

### Scope for Improvement

- **Multimodal learning:** Combining titles with book cover images or summaries could provide richer features and boost accuracy.
- **Handling multi-label genres:** Some books fit multiple genres, but current models assume only one label per book.
- **Data balance:** Addressing class imbalance could help underrepresented genres.

---



