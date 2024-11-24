from sklearn.svm import SVC
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle

def classifier_knn(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, knn, X_test, y_test

def classifier_rf(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, rf, X_test, y_test

def classifier_svm(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', C=1, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, svm, X_test, y_test
def classifier_dt(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, dt, X_test, y_test

def classifier_lr(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, lr, X_test, y_test

def classifier_gb(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, gb, X_test, y_test

def save_classifier(classifier, filename):
    with open(filename, 'wb') as file:
        pickle.dump(classifier, file)

def train_and_save_classifiers(classifiers, feature_matrix):
    for name, classifier_func in classifiers.items():
        accuracy, _, classifier, _, _ = classifier_func(feature_matrix)
        save_classifier(classifier, f"../Generated_Files/{name}_classifier.pkl")
        print(f"{name} classifier trained and saved.")
if __name__ == '__main__':
    classifiers = {
        "KNN": classifier_knn,
        "Random Forest": classifier_rf,
        "SVM": classifier_svm,
        "Logistic Regression": classifier_lr,
        "Decision Tree": classifier_dt,
        "Gradient Boosting": classifier_gb,
    }
    feature_matrix = pd.read_csv('../Generated_Files/matrix.csv')
    train_and_save_classifiers(classifiers,feature_matrix)

