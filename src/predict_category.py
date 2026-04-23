import joblib
import pandas as pd


def main():
    model = joblib.load("model/category_model.pkl")

    print("Model loaded successfully!")
    print("Type 'exit' at any point to stop.\n")

    while True:
        title = input("Enter product title: ").strip()

        if title.lower() == "exit":
            print("Exiting...")
            break

        # Feature engineering for user input
        user_input = pd.DataFrame([{
            "Product Title": title,
            "title_length_chars": len(title),
            "title_word_count": len(title.split()),
            "has_digits": int(any(char.isdigit() for char in title))
        }])

        prediction = model.predict(user_input)[0]
        print(f"Predicted category: {prediction}")
        print("-" * 50)


if __name__ == "__main__":
    main()