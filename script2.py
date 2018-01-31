from sklearn import metrics
from sklearn.externals import joblib

from modules import normalize_input

print("Enter model file address:")
model_file_address = input()
print("Enter test content file address:")
test_content_file_address = input()
print("Enter test label file address:")
test_label_file_address = input()

test_data = normalize_input(test_content_file_address, test_label_file_address)
model = joblib.load(model_file_address)
predicted = model.predict(test_data.data)

file = open("results.txt", "w")
for a_predicted in predicted:
    file.write(str(a_predicted) + '\n')
file.close()
print("The results are saved in results.txt")

print(metrics.classification_report(test_data.target, predicted))
print(metrics.confusion_matrix(test_data.target, predicted))
