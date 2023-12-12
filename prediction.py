import joblib


def predict(data):
    clf = joblib.load("modelLab1.sav")
    return clf.predict(data)
