import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main():
    df = pd.read_csv("data/IMLP4_TASK_03-products.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Keep only relevant columns
    df = df[["Product Title", "Category Label"]].copy()

    # Drop missing values in essential columns
    df = df.dropna(subset=["Product Title", "Category Label"])

    # Standardize text
    df["Product Title"] = df["Product Title"].astype(str).str.strip()
    df["Category Label"] = df["Category Label"].astype(str).str.strip()

    # Feature engineering
    df["title_length_chars"] = df["Product Title"].astype(str).str.len()
    df["title_word_count"] = df["Product Title"].astype(str).apply(lambda x: len(x.split()))
    df["has_digits"] = df["Product Title"].astype(str).str.contains(r"\d").astype(int)

    # Features and label
    X = df[["Product Title", "title_length_chars", "title_word_count", "has_digits"]]
    y = df["Category Label"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", TfidfVectorizer(), "Product Title"),
            ("numeric", MinMaxScaler(), ["title_length_chars", "title_word_count", "has_digits"])
        ]
    )

    # Final model
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LinearSVC())
    ])

    # Train on full dataset
    model.fit(X, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/category_model.pkl")

    print("Model trained and saved as 'model/category_model.pkl'")


if __name__ == "__main__":
    main()