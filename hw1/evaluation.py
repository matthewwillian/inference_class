from collections import Counter

import funcy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss, confusion_matrix


def parse_dataset(filename):
    dataset_file = open(filename)
    X = []
    y = [] 
    for line in dataset_file:
        tokens = line.split()
        is_spam = int(tokens[1] == "spam")
        word_counts = {}
        for token, count in funcy.partition(2, tokens[2:]):
            word_counts[token] = int(count)
        X.append(word_counts)
        y.append(is_spam)
    return X, y


def calculate_class_priors(y):
    return funcy.walk_values(lambda v: v / len(y), dict(Counter(y)))


def m_estimate(n_c, p, m, n):
    return (n_c + p * m) / (m + n)


def sum_word_vectors(v):
    return funcy.merge_with(sum, *v)


def calculate_m_estimates(X, y, alpha=1.0):
    total_counts = sum_word_vectors(X)
    vocabulary = total_counts.keys()
    m = len(vocabulary) * alpha
    p = 1 / len(vocabulary)
    n = sum(total_counts.values())

    grouped_counts = {
        k: sum_word_vectors(v)
        for k, v in funcy.group_values(zip(y, X)).items()
    }
    counts_as_series = {
        k: pd.Series(v).reindex(vocabulary, fill_value=0) 
        for k, v in grouped_counts.items()
    }
    likelihood_m_estimates = {
        k: m_estimate(v, p, m, n)
        for k, v in counts_as_series.items()
    }

    return likelihood_m_estimates


class Classifier:
    def __init__(self, alpha):
        self.alpha = alpha

    def train(self, X, y):
        likelihood = calculate_m_estimates(X, y, self.alpha)
        self.log_likelihood = funcy.walk_values(np.log, likelihood)
        self.log_class_priors = funcy.walk_values(np.log, calculate_class_priors(y))

    def _predict_class_posterior(self, x, y):
        x_series = pd.Series(x)

        return self.log_class_priors[y] + np.sum(self.log_likelihood[y] * x_series)

    def predict(self, x):        
        return max(self.log_class_priors.keys(),
                   key=lambda y: self._predict_class_posterior(x, y))

X_train, y_train = parse_dataset("train")

# # Spam percentage
# print("P(spam) = {}".format(sum(y_train) / len(y_train)))
# 
# # Most frequent words
# m_estimates = calculate_m_estimates(X_train, y_train)
# print("Top 5 ham words")
# print(m_estimates[0].sort_values(ascending=False).head())
# print("Top 5 spam words")
# print(m_estimates[1].sort_values(ascending=False).head())

X_test, y_test = parse_dataset("test")

def test_alpha(alpha):
    classifier = Classifier(alpha)
    classifier.train(X_train, y_train)

    y_predicted = [classifier.predict(x) for x in X_test]
    print("Loss for alpha {} is {}".format(alpha, zero_one_loss(y_test, y_predicted)))
    print(sum(y_test))
    print(sum(y_predicted))
    print(confusion_matrix(y_test, y_predicted))

# Alter alpha parameters

errors = []
alphas = []
for n in range(-2, 5):
    classifier = Classifier(10 ** n)
    classifier.train(X_train, y_train)

    y_predicted = [classifier.predict(x) for x in X_test]
    errors.append(zero_one_loss(y_test, y_predicted)) 
    alphas.append(10 ** n)

m = np.array(alphas) * len(sum_word_vectors(X_train))
plt.semilogx(m, errors)
plt.xlabel("m")
plt.ylabel("% error")
plt.savefig("error_plot.png")

