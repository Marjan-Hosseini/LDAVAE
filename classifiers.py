import sys
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sys.path.append('../')


def compute_accuracy_metrics(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return accuracy, precision, recall, fscore, fpr, fnr


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def svm_linear(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    svm_clf4 = SVC(kernel="linear")
    svm_clf4.fit(x_train, y_train)
    y_pred = svm_clf4.predict(x_test)
    y_pred_tr = svm_clf4.predict(x_train)

    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def logistic_regression(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)
    y_pred_tr = lr_model.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def mlp_classifier(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_tr = clf.predict(x_train)

    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def random_forest_classifier(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_tr = clf.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def naive_bayes(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    y_pred_tr = gnb.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def svm_nonlinear(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    svm_model = SVC(kernel ='rbf')
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)
    y_pred_tr = svm_model.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def knn3(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    y_pred_tr = knn_model.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)


def linear_discriminat_analysis(x_train, x_test, y_train, y_test):
    x_train, x_test = scale_data(x_train, x_test)
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(x_train, y_train)
    y_pred = lda_model.predict(x_test)
    y_pred_tr = lda_model.predict(x_train)
    return compute_accuracy_metrics(y_train, y_pred_tr), compute_accuracy_metrics(y_test, y_pred)
