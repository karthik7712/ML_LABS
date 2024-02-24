import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc, f1_score,precision_score,recall_score
from performance_metrics import PerformanceMetrics


def naive_bayes(dataset):
    for x in range(1,6):
        clf = GaussianNB()
        train_dataset = []
        for i in range(1,6):
            if i != x:
                train_data = pd.read_csv(f"./{dataset}-5-fold/{dataset}-5-{i}tra.csv")
                train_dataset.append(train_data)

        train_dataset = pd.concat(train_dataset,ignore_index=True)
        test_dataset = pd.read_csv(f"./{dataset}-5-fold/{dataset}-5-{x}tst.csv")
        x_train = train_dataset.iloc[:,:-1].values
        y_train = train_dataset.iloc[:,-1].values

        x_test = test_dataset.iloc[:,:-1].values
        y_test = test_dataset.iloc[:,-1].values

        clf.fit(x_train,y_train)

        y_pred = clf.predict(x_test)
        print(f"The results for {x} fold:")

        print("Performance metrics using library functions:")
        accuracy = accuracy_score(y_test,y_pred)
        conf_mat = confusion_matrix(y_test,y_pred)
        precision =  precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        fmeas = f1_score(y_test,y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        far = 1 - tpr[1]
        roc_auc = auc(fpr, tpr)
        # print(tpr)
        print("Confusion matrix:")
        print(conf_mat)
        print("Overall Accuracy:",round(accuracy,2))
        print("Precision:",round(precision),2)
        print("Recall:",round(recall,2))
        print("True Positive rate:",round(tpr[1],2))
        print("False Alarming rate:",round(far,2))
        print("fmeasure:", round(fmeas,2))
        print("Area under roc curve:",round(roc_auc,2))
        print("\n")
        print("Performance metrics without using library functions:")
        evaluate = PerformanceMetrics(y_test,y_pred)
        print("Confusion matrix:")
        print(evaluate.confusion_matrix())
        print("Overall Accuracy:",round(evaluate.accuracy(),2))
        print("Precision:",round(evaluate.precision(),2))
        print("Recall:",round(evaluate.recall(),2))
        print("True Positive rate:",round(evaluate.true_positive_rate(),2))
        print("False Alarming rate:",round(evaluate.false_alarming_rate(),2))
        print("fmeasure:", round(evaluate.fmeasure(),2))
        print("Area under roc curve:",round(evaluate.area_under_roc(),2))
        print("\n\n")


naive_bayes(dataset="titanic")








