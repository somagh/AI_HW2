# IN THE NAME OF ALLAH
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

# from hazm import *


def normalize_input(train_content_file_address, train_label_file_address):
    content_file = open(train_content_file_address, "r", encoding="utf8")
    label_file = open(train_label_file_address, "r")
    data = Bunch(data=[], target=[])
    # normalizer = Normalizer()
    # stemmer = Stemmer()
    # lemmatizer = Lemmatizer()

    for line in content_file:
        # print(line)
        # text = normalizer.normalize(line)
        # arr = word_tokenize(text)
        # arr_stemmize = [stemmer.stem(x) for x in arr]
        # arr_lemmatize = [lemmatizer.lemmatize(x) for x in arr]
        # final_lemmatize = " ".join(arr_lemmatize)
        # final_stemmize = " ".join(arr_stemmize)
        # print(final_stemmize)
        # print(final_lemmatize)
        # sys.exit()
        data.data.append(line)

    for line in label_file:
        clas = line.rstrip()
        data.target.append(clas)

    return data

def create_model():
    return Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), (
    'clf', SGDClassifier(loss='hinge', penalty='l1', alpha= 1e-5, random_state=42, max_iter=5, tol=None))])