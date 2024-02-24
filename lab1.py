import csv
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd

setlist = ['diabetes','baseball','friedman','laser','machineCPU','plastic']

max_r = (10,0)
min_rs =(10,sys.maxsize)
min_a = (10,sys.maxsize)
al = 0


#Perform different Regressions on the datasets mentioned above in a 5-fold cross validation method


def convert_to_csv(d, x):
    with open(f'./{d}-5-fold/{d}-5-{x}tra.dat', 'r') as file:
        lines = file.readlines()

    # Extract attribute names and data
    attributes = []
    data = []

    # Parse the lines
    for line in lines:
        line = line.strip()
        if line.startswith('@attribute'):
            attribute_name = line.split()[1]
            attributes.append(attribute_name)
        elif line.startswith('@data'):
            continue
        elif not line.startswith('@'):
            data.append(line.split(','))

    # Write to CSV
    with open(f'./{d}-5-fold/{d}-5-{x}tra.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(attributes)

        # Write the data rows
        writer.writerows(data)

    with open(f'./{d}-5-fold/{d}-5-{x}tst.dat', 'r') as file:
        lines = file.readlines()

    # Extract attribute names and data
    attributes = []
    data = []

    # Parse the lines
    for line in lines:
        line = line.strip()
        if line.startswith('@attribute'):
            attribute_name = line.split()[1]
            attributes.append(attribute_name)
        elif line.startswith('@data'):
            continue
        elif not line.startswith('@'):
            data.append(line.split(','))

    # Write to CSV
    with open(f'./{d}-5-fold/{dataset}-5-{x}tst.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(attributes)

        # Write the data rows
        writer.writerows(data)


def linear_regression(dataset,):
    global max_r,min_a,min_rs
    accuracy = []
    mae = []
    rmse = []
    r2 = []
    for i in range(1,6):
        model = LinearRegression()
        training_datasets = []
        for x in range(1,6):
            if x != i:
                df_train = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{x}tra.csv')
                training_datasets.append(df_train)
                # print(df_train)
        training_dataset = pd.concat(training_datasets,ignore_index=True)
        testing_dataset = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{i}tst.csv')
        # print(training_dataset)
        x_train = training_dataset.iloc[:, :-1].values
        y_train = training_dataset.iloc[:, -1].values

        x_test = testing_dataset.iloc[:, :-1].values
        y_test = testing_dataset.iloc[:, -1].values

        # imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        m = mean_absolute_error(y_test,y_pred)
        rm = np.sqrt(m)
        r2 = r2_score(y_test,y_pred)
        ms = mean_squared_error(y_test,y_pred)


        mae.append(m)
        rmse.append(rm)
        accuracy.append(r2*100)

    errors = {'RMSE':rmse,'MAE':mae,'Accuracy':accuracy}

    rs = min(errors['RMSE'])
    min_rs = (1,rs)

    r = min(errors['Accuracy'])
    if r>0:
        max_r = (1,r)

    s = min(errors['MAE'])
    min_a = (1,s)

    data = pd.DataFrame(errors)
    print(data)
    avg_accuracy = sum(accuracy)/len(accuracy)
    print("Average accuracy :", round(avg_accuracy,2))


def poly_reg(dataset,k):
    global max_r,min_a,min_rs
    accuracy = []
    mae = []
    rmse = []
    r2 = []
    for i in range(1,6):
        test_number = i
        model = LinearRegression()
        training_datasets = []
        for x in range(1,6):
            if x != i:
                df_train = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{x}tra.csv')
                training_datasets.append(df_train)
        training_dataset = pd.concat(training_datasets,ignore_index=True)
        testing_dataset = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{i}tst.csv')
        x_train = training_dataset.iloc[:, :-1].values
        y_train = training_dataset.iloc[:, -1].values

        x_test = testing_dataset.iloc[:, :-1].values
        y_test = testing_dataset.iloc[:, -1].values

        poly_features = PolynomialFeatures(degree=k)
        x_train_poly = poly_features.fit_transform(x_train)
        x_test_poly = poly_features.fit_transform(x_test)

        model.fit(x_train_poly,y_train)

        y_pred = model.predict(x_test_poly)

        m=mean_absolute_error(y_test,y_pred)
        rm = np.sqrt(m)
        r2 = r2_score(y_test,y_pred)

        mae.append(m)
        rmse.append(rm)
        accuracy.append(r2 * 100)

    errors = {'RMSE': rmse, 'MAE': mae, 'Accuracy': accuracy}

    r = min(errors['Accuracy'])
    if max_r[1] < r :
        max_r = (k,r)

    a = min(errors['MAE'])
    if min_a[1] > a:
        min_a = (k,a)

    rs = min(errors['RMSE'])
    if min_rs[1] > rs:
        min_rs = (k,rs)


    data = pd.DataFrame(errors)
    print(data)
    avg_accuracy = sum(accuracy)/len(accuracy)
    print("Average accuracy:",round(avg_accuracy,2))


def ridge_regression(dataset):
    global al
    all_rmse = []
    all_mse = []
    all_accuracy = []

    for i in range(1, 6):
        training_datasets = []
        alphas = 2.0 ** np.arange(-18, 51, 2)

        for x in range(1, 6):
            if x != i:
                df_train = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{x}tra.csv')
                training_datasets.append(df_train)

        training_dataset = pd.concat(training_datasets, ignore_index=True)
        testing_dataset = pd.read_csv(f'./{dataset}-5-fold/{dataset}-5-{i}tst.csv')
        x_train = training_dataset.iloc[:, :-1].values
        y_train = training_dataset.iloc[:, -1].values

        x_test = testing_dataset.iloc[:, :-1].values
        y_test = testing_dataset.iloc[:, -1].values

        rmse_fold = []
        mse_fold = []
        r2_fold = []

        for alpha in alphas:
            ridge_reg = Ridge(alpha=alpha)
            ridge_reg.fit(x_train, y_train)

            y_pred = ridge_reg.predict(x_test)
            m = mean_absolute_error(y_test, y_pred)
            rm = np.sqrt(m)
            r = r2_score(y_test, y_pred)

            rmse_fold.append(rm)
            mse_fold.append(m)
            r2_fold.append(r)

        optimal_alpha_index = np.argmin(rmse_fold)
        al = alphas[optimal_alpha_index]

        rmse_fold_optimal = rmse_fold[optimal_alpha_index]
        mse_fold_optimal = mse_fold[optimal_alpha_index]
        r2_fold_optimal = r2_fold[optimal_alpha_index]

        all_rmse.append(rmse_fold_optimal)
        all_mse.append(mse_fold_optimal)
        all_accuracy.append(r2_fold_optimal * 100)

    avg_accuracy = sum(all_accuracy) / len(all_accuracy)

    errors = {'RMSE': all_rmse, 'MSE': all_mse, 'Accuracy': all_accuracy}
    data = pd.DataFrame(errors)

    print(data)
    print("Average Accuracy:", round(avg_accuracy,2))


for dataset in setlist:
    # for x in range(1,6):
    #     convert_to_csv(dataset,x)



    print(f"Linear Regression metrics for {dataset} and its accuracy is ")
    linear_regression(dataset)
    print(f"Polynomial Regression metrics for {dataset} and its accuracy is (order=2)")
    poly_reg(dataset,2)
    print(f"Polynomial Regression metrics for {dataset} and its accuracy is (order=3)")
    poly_reg(dataset,3)
    print(f"Ridge Regression metrics for {dataset} and its accuracy is")
    ridge_regression(dataset)

    print("best order an value for cod is",max_r)
    print("best order an value for rmse is",min_rs)
    print("best order an value for mae is",min_a)
    print("Regularization Parameter is",al)

# import numpy as np
#
# class LinearRegressionGD:
#     def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0):
#         self.learning_rate = learning_rate
#         self.n_iterations = n_iterations
#         self.alpha = alpha  # Ridge regularization parameter
#         self.weights = None
#         self.bias = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         for _ in range(self.n_iterations):
#             # Predictions
#             y_pred = np.dot(X, self.weights) + self.bias
#
#             # Gradient of loss function w.r.t. weights and bias
#             d_weights = (1/n_samples) * (2 * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights)
#             d_bias = (1/n_samples) * np.sum(2 * (y_pred - y))
#
#             # Update weights and bias
#             self.weights -= self.learning_rate * d_weights
#             self.bias -= self.learning_rate * d_bias
#
#     def predict(self, X):
#         return np.dot(X, self.weights) + self.bias
#
# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred)**2)
#
# def mean_absolute_error(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))
#
# def r2_score(y_true, y_pred):
#     numerator = np.sum((y_true - y_pred)**2)
#     denominator = np.sum((y_true - np.mean(y_true))**2)
#     return 1 - (numerator / denominator)
#
#
# X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
# y = np.array([3, 4, 5, 7, 5])
# learning_rate = 0.01
# n_iterations = 1000
# alpha = 0.1
# model = LinearRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, alpha=alpha)
# model.fit(X, y)
# y_pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# mae = mean_absolute_error(y, y_pred)
# r2 = r2_score(y, y_pred)
#
# print("RMSE:", rmse)
# print("MAE:", mae)
# print("R-squared:", r2)