from sklearn.externals import joblib

from modules import normalize_input, create_model

print("Enter train content file address:")
train_content_file_address=input()
print("Enter train label file address:")
train_label_file_address=input()

learn_data = normalize_input(train_content_file_address,train_label_file_address)
model = create_model()
model.fit(learn_data.data,learn_data.target)
joblib.dump(model, "model.txt")

print("The model is saved in model.txt")