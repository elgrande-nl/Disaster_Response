# In a Python script, train_classifier.py, write a machine learning pipeline that:

# Loads data from the SQLite database
# Splits the dataset into training and test sets
# Builds a text processing and machine learning pipeline
# Trains and tunes a model using GridSearchCV
# Outputs results on the test set
# Exports the final model as a pickle file


import logging
import sys
import re
import numpy as np
import pandas as pd
import pickle
import sqlite3
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)


nltk.download(["wordnet", "punkt", "stopwords"])


def load_data(database_filepath):
    """
    Function:
    Load data from the database

    Args:
        database_filepath (str): the path of the database containing the data for the model

    Returns:
        X (DataFrame): Table that contains the messages
        Y (DataFrame): the classification table
        category_names (list): list of the categorie names
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("tbl_MessagesCategories", engine)

    X = df["message"]  # message
    Y = df.iloc[:, 5:]  # Classifications

    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Function:
    To clean the dataset of urls and split the input text into meaningfull words.

    Args:
        text (str): the message

    Returns:
        tokens(list of str) a list of meaningfull words.
    """
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # detect and remove urls from text.
    found_urls = re.findall(url_regex, text)

    for url in found_urls:
        text = text.replace(url, "weblink_placeholder")

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]

    return tokens


def build_model():
    """
    Function:
    build the model and do a gridsearch

    Returns:
        cv: {Sklearn GridSearchCV}: Gridsearch model object
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    # Create Grid search parameters
    parameters = {
        "tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [10, 50],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function:
    prints the classification results of the model based on the Test set.

    Args:
        model (_type_): the scikit-learn fitted model
        X_test (_type_): The X test set
        Y_test (_type_): the Y test classifications set
        category_names (_type_): the category names
    """

    Y_pred = model.predict(X_test)

    for i in range(36):
        print("=======================", Y_test.columns[i], "======================")
        print(
            classification_report(
                Y_test.iloc[:, i], Y_pred[:, i], target_names=category_names
            )
        )


def save_model(model, model_filepath):
    """
    Function:
    To save the model as a pickel.

    Args:
        model : The classification model
        model_filepath (str): the path to the pickelfile.
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    print(sys.argv)
    main()
