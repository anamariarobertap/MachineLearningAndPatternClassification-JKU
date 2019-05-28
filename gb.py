import os

from scipy.io import arff
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier

TRAINING_FILES_NO = [3, 4, 5, 6, 9, 12, 13, 14]
TEST_FILES_NO = [1, 2, 10]
VALIDATION_FILES_NO = [7, 8, 11]



def load_data_from_list_of_files(data_type, list_of_files=None):
    if list_of_files is None:
        list_of_files = TRAINING_FILES_NO
    dfl = []
    for i in list_of_files:
        dfl.append(load_data_from_specific_file_no(data_type=data_type, file_no=i))
    df = pd.concat(dfl)
    return df


def load_training_data(data_type):
    return load_data_from_list_of_files(data_type, TRAINING_FILES_NO)


def load_test_data(data_type):
    return load_data_from_list_of_files(data_type, TEST_FILES_NO)


def load_validation_data(data_type):
    return load_data_from_list_of_files(data_type, VALIDATION_FILES_NO)


def load_data_from_specific_file_no(
        root_dir=r'D:\JKU\ML PatternClassification\FirstTry\train',
        data_type='music', file_no=1):
    """
    This function will load data from all the files 1.<data_type>.arff to <how_many>.<data_type>.arff
    By default will load music training instances from all the files 1..14
    :param file_no: file no to load
    :param root_dir: dir to path where arff files containing data are stored
    :param data_type: str: 'music' or 'speech'
    :return: a pandas dataframe containing loaded data
    """

    data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, file_no, data_type))
    df = pd.DataFrame(data[0])
    return df


def load_data(root_dir=r'D:\JKU\ML PatternClassification\FirstTry\train',
              data_type='music', how_many=14, last=False):
    """
    This function will load data from all the files 1.<data_type>.arff to <how_many>.<data_type>.arff
    By default will load music training instances from all the files 1..14
    :param root_dir: dir to path where arff files containing data are stored
    :param data_type: str: 'music' or 'speech'
    :param how_many: 'number of files to load' ,
    :return: a pandas dataframe containing loaded data
    """

    dfl = []
    if not last:
        for i in range(1, how_many + 1):
            data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, i, data_type))
            df = pd.DataFrame(data[0])
            dfl.append(df)
    else:
        for i in range(14, 14 - how_many, -1):
            data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, i, data_type))
            df = pd.DataFrame(data[0])
            dfl.append(df)
    df = pd.concat(dfl)
    return df


# a = load_validation_data('speech')
# print(a.shape())

def gradientBoostingClassifier():
    gb = GradientBoostingClassifier()
    data_train = load_training_data("music")
    data_test = load_test_data("music")
    data_validation = load_validation_data("music")
    y_test = data_test.values[:, -1].astype(str)
    y_train = data_train.values[:,-1].astype(str)
    y_validation = data_validation.values[:,-1].astype(str)

    X_train = data_train.values[:, :-1].astype(str)
    X_test = data_test.values[:, :-1].astype(str)
    X_validation = data_validation.values[:, :-1].astype(str)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_validation_scale = scaler.transform(X_validation)

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

    learning_rates = [0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in learning_rates:
        gb = GradientBoostingClassifier(n_estimators=60, learning_rate=learning_rate,max_depth=5)
        gb = gb.fit(X_train_scale, y_train)
        #print("Learning rate: ", learning_rate)
        #rint("Accuracy score (training): {0:.3f}".format(gb.score(X_train_scale, y_train)))
        #print("Accuracy score (validation): {0:.3f}".format(gb.score( X_validation_scale, y_validation)))
        #print()
    #y_scores_gb = gb.decision_function(data_train)
    #fpr_gb, tpr_gb, _ = roc_curve(y_validation, y_scores_gb)
    #roc_auc_gb = auc(fpr_gb, tpr_gb)
    from joblib import dump
    dump(gb, 'D:\\JKU\\ML PatternClassification\\models\\modelIIImu.joblib')
    #print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

gradientBoostingClassifier()
