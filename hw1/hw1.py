# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    max_X = X.max(axis=0)
    mean_X = X.mean(axis=0)
    min_X = X.min(axis=0)
    max_y = y.max()
    mean_y = y.mean()
    min_y = y.min()
    X = (X - mean_X) / (max_X - min_X)
    y = (y - mean_y) / (max_y - min_y)

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ones_column = np.ones(X.shape[0])
    X = np.c_[ones_column, X]
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    X_theta = np.dot(X, theta)
    J = np.sum((X_theta - y) ** 2) / (2 * X.shape[0])

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - alpha * 1 / X.shape[0] * np.dot(X.T, h - y)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_theta = []
    X_transpose = np.transpose(X)
    X_pinv = np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose)
    pinv_theta = np.dot(X_pinv, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - (alpha * np.dot(X.T, h - y)) / X.shape[0]
        J_history.append(compute_cost(X, y, theta))
        if i > 0 and J_history[i - 1] - J_history[i] < 1e-8:
            break

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    for alpha in alphas:
        theta = np.ones(X_train.shape[1])
        theta, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    X_features_included_train = np.ones(X_train.shape[0])
    X_features_included_val = np.ones(X_val.shape[0])
    # feature_names = df.columns.values
    for i in range(5):
        min_loss = float('inf')
        min_feature = -1
        # feature_dict = {}
        for feature in range(X_train.shape[1]):
            if feature in selected_features:
                continue
            selected_features.append(feature)
            theta = np.ones(len(selected_features) + 1)
            X_temp_train = np.c_[X_features_included_train, X_train[:, feature]]
            X_temp_val = np.c_[X_features_included_val, X_val[:, feature]]
            theta, J_history = efficient_gradient_descent(X_temp_train, y_train, theta, best_alpha, iterations)
            loss = compute_cost(X_temp_val, y_val, theta)
            # feature_dict[feature_names[feature]] = loss
            if loss < min_loss:
                min_loss = loss
                min_feature = feature
            selected_features.remove(feature)

        selected_features.append(min_feature)
        X_features_included_train = np.c_[X_features_included_train, X_train[:, min_feature]]
        X_features_included_val = np.c_[X_features_included_val, X_val[:, min_feature]]
        # print(f'Added feature {feature_names[min_feature]} with loss {min_loss}')
        # print(dict(sorted(feature_dict.items(), key=lambda item: item[1])))

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    poly_features = []
    num_columns = len(df.columns)
    for i in range(num_columns):
        for j in range(i, num_columns):
            column1 = df.columns[i]
            column2 = df.columns[j]
            poly_feature_name = column1 + '*' + column2
            poly_feature_values = df[column1] * df[column2]
            poly_features.append(pd.Series(poly_feature_values, name=poly_feature_name))

    # Concatenate all polynomial features to the original dataframe
    df_poly = pd.concat([df_poly] + poly_features, axis=1)

    return df_poly
