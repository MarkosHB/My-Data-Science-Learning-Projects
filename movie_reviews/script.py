# %%
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random

# %%
# Download the dataset.
nltk.download('movie_reviews')

# Adjunct the review words to its category.
reviews = [(list(word.lower() for word in movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Ramdomize the reviews order.
random.shuffle(reviews)

# Print the first sample.
print(reviews[0]) 

# %%
# Separate the words and categories for training.
words, labels = zip(*reviews)

# Efectuate the partition of the dataset into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(words, labels, test_size=0.2, random_state=42)

# Determine what words are present.
def is_present(words):
    return {word: True for word in words}

x_train_features = [(is_present(words), label) for words, label in zip(x_train, y_train)]
x_test_features = [(is_present(words), label) for words, label in zip(x_test, y_test)]

# %%
# Train the Naive Bayes classifier.
model = NaiveBayesClassifier.train(x_train_features)

# Make predictions on the test set.
y_pred = [model.classify(features) for features, _ in x_test_features]

# Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


