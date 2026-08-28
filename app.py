
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("sentiment_data.csv")

# Separate text and sentiment
texts = data["text"]
labels = data["sentiment"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Streamlit UI
st.title("Sentiment Analysis")

st.write(
    "Enter a piece of text and the model will predict its sentiment."
)

user_text = st.text_area(
    "Enter your text:",
    placeholder="e.g. I absolutely loved this product!"
)

if st.button("Analyse Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Convert input into TF-IDF features
        text_vector = vectorizer.transform([user_text])

        # Predict sentiment
        prediction = model.predict(text_vector)[0]

        # Get confidence
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction.lower() == "positive":
            st.success(
                f"Positive — {confidence:.1f}% confidence"
            )
        else:
            st.error(
                f"Negative — {confidence:.1f}% confidence"
            )
