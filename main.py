# in the name of god
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from hazm import *


data = load_files('./sath', encoding='utf-8')

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(data.data)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, data.target)
# joblib.dump(clf, "model.txt")
# clf2 = joblib.load("model.txt")
#
# print(clf2)
# print(clf)
# print(X_train_tfidf)
print(data)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf.fit(data.data, data.target)
joblib.dump(text_clf, "model.txt")

predicted = text_clf.predict(data.data)

acc = np.mean(predicted == data.target)

scores = cross_val_score(text_clf, data.data, data.target, cv=2)


print(np.array([2,3]).mean())
print(acc)
print(metrics.classification_report(data.target, predicted, target_names=data.target_names))
print(metrics.confusion_matrix(data.target, predicted))

print(text_clf)




# normalizer = Normalizer()
# stemmer = Stemmer()
# arr = word_tokenize(data.data[1])
# arr2 = [stemmer.stem(x) for x in arr]
# print(word_tokenize(data.data[1]))


