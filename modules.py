# IN THE NAME OF ALLAH
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch


def normalize_input(train_content_file_address, train_label_file_address):
    content_file = open(train_content_file_address, "r", encoding="utf8")
    label_file = open(train_label_file_address, "r")
    data = Bunch(data=[], target=[])

    for line in content_file:
        data.data.append(line)

    for line in label_file:
        clas = line.rstrip()
        data.target.append(clas)

    return data

def create_model():
    return Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), (
    'clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])