# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

def eval_model(model,x_test,y_test):
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label="Predicted vs Actual")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    return {
        "Mean Squared Error": mse,
        "R-squared": r2,
    }

def normalize_rows(df):
    return df.div(df.sum(axis=1), axis=0).dropna()