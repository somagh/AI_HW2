# IN THE NAME OF ALLAH

from sklearn.utils import Bunch


def normalize_data_files(train_content_file_address, train_label_file_address):
    content_file = open(train_content_file_address, "r", encoding="utf8")
    label_file = open(train_label_file_address, "r")
    data = Bunch(data=[], target=[], target_names=[])

    for line in content_file:
        data.data.append(line)

    for line in label_file:
        clas = line.rstrip()
        if not clas in data.target_names:
            data.target_names.append(clas)
        data.target.append(data.target_names.index(clas))

    return data
