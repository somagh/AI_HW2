# IN THE NAME OF ALLAH

import os
import glob


def normalize_data_files(train_content_file_address, train_label_file_address):
    content_file = open(train_content_file_address, "r", encoding="utf8")
    label_file = open(train_label_file_address, "r")
    class_counter = {}

    contents = []
    labels = []
    label = "vez"

    for line in content_file:
        contents.append(line)

    for line in label_file:
        clas = line.rstrip()
        labels.append(clas)
        if not (clas in class_counter):
            class_counter[clas] = 0

    path = "./data"

    # files = glob.glob(path + "\*")
    #
    # for file in files:
    #     print(file)
    #     os.remove(file)


    for class_number in labels:
        new_path = path + "\class" + class_number
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    for i in range(len(contents)):
        class_counter[labels[i]] = class_counter[labels[i]] + 1
        file = open(path + "\class" + labels[i] + "\sample" + str(class_counter[labels[i]]) + ".txt", "w",
                    encoding="utf8")
        file.write(contents[i])
        file.close()

    print("Normalized data files are ready !!")

    return path
