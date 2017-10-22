import sklearn
from sklearn import datasets
from os import listdir
from os.path import isfile, join
import io
from sklearn import svm
import collections
import Baseline_NB


def predict_classify(category_path, test_path):
    ip_data = sklearn.datasets.load_files(category_path)
    ip_test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    ip_test = []
    op_test = collections.OrderedDict()
    for test in ip_test_files:
        ip = io.open(path_test + '\\' + test, 'r', encoding='utf8')
        for ip_line in ip.readlines():
            ip_test.append(ip_line)
            op_test[str(ip_line.split(None, 1)[0])] = ""

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))
    training_vector = vectorizer.fit_transform(ip_data.data)
    print(training_vector)
    test_vector = vectorizer.transform(ip_test)

    classifier = svm.LinearSVC()
    classifier.fit(training_vector, ip_data.target)
    prediction = classifier.predict(test_vector)

    for i, (key, value) in enumerate(op_test.items()):
        if 1 == prediction[i]:
            op_test[key] = 'POS'
        else:
            op_test[key] = 'NEG'

    return op_test


if __name__ == "__main__":

    path_ip = 'C:\\Users\\Amruta\\PycharmProjects\\SA_BSF\\category_2'
    path_test = 'C:\\Users\\Amruta\\PycharmProjects\\SA_BSF\\test_2'
    gold_pos = r'hotelPosT-train.txt'
    gold_neg = r'hotelNegT-train.txt'
    dev_set = r'test_doc.txt'
    output_file = 'output_SVM.txt'
    verdict = predict_classify(path_ip, path_test)
    accuracy = Baseline_NB.give_accuracy(gold_pos, gold_neg, verdict)
    Baseline_NB.print_output(verdict, output_file, dev_set)
    print(verdict)
    print(accuracy)






