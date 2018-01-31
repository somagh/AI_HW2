from sklearn.model_selection import cross_val_score

from modules import normalize_input, create_model

print("Enter content file address:")
content_file_address = input()
print("Enter label file address:")
label_file_address = input()

data = normalize_input(content_file_address, label_file_address)
model = create_model()
scores = cross_val_score(model, data.data, data.target, cv=5)
print("The model accuracy is " + str(scores.mean()))
