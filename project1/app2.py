from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the data
data = pd.read_csv('news heading.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Category'], test_size=0.2)

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the CountVectorizer to the training data
vectorizer.fit(X_train)

# Transform the training and test data
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# Create a MultinomialNB object
classifier = MultinomialNB()

# Fit the MultinomialNB to the training data
classifier.fit(X_train, y_train)

# Define an endpoint for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        predicted_category = predict(headline)
        return render_template('index.html', headline=headline, predicted_category=predicted_category)
    return render_template('index.html', headline=None, predicted_category=None)

# Define a function for making predictions
def predict(headline):
    headline_vector = vectorizer.transform([headline])
    predicted_category = classifier.predict(headline_vector)
    return predicted_category[0]

if __name__ == '__main__':
    app.run(debug=True)
