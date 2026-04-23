# ML Product Category Classification

A machine learning project for automatic product category classification based on product titles.

## Project structure

- `data/` - dataset used for training
- `notebooks/` - analysis and experimentation notebooks
- `model/` - saved trained model
- `src/` - Python scripts for training and prediction

## Dataset

The dataset used in this project is stored in:

`data/IMLP4_TASK_03-products.csv`

Main relevant columns:
- `Product Title`
- `Category Label`

## Models tested

In this project, I compared the following machine learning models:
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC

The best-performing model was **Linear SVC**, with an accuracy of approximately **96.67%**.

## Feature engineering

The following features were used:
- TF-IDF representation of product titles
- title length in characters
- title word count
- whether the title contains digits

## Files

### `src/train_model.py`
This script trains the final model on the full dataset and saves it as:

`model/category_model.pkl`

### `src/predict_category.py`
This script loads the saved model and allows interactive category prediction based on a product title entered by the user.

## How to run

### Train the model

```bash
python src/train_model.py
