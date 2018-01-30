from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from InputNormalizer import normalize_data_files


print("Enter train content file address:")
train_content_file_address=input()
print("Enter train label file address:")
train_label_file_address=input()

data = normalize_data_files(train_content_file_address,train_label_file_address)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf.fit(data.data, data.target)
joblib.dump(text_clf, "model.txt")

print("The model is saved in model.txt")