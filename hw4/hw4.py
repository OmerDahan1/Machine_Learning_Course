import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    denominator_x = 0.0
    denominator_y = 0.0
    for i in range(len(x)):
        r += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        denominator_x += (x[i] - np.mean(x)) ** 2
        denominator_y += (y[i] - np.mean(y)) ** 2

    denominator = np.sqrt(denominator_x * denominator_y)
    r = r / denominator

    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    X_numeric = X.select_dtypes(include=[np.number])

    for i in range(n_features):
        correlation = 0.0
        best_feature = None
        for feature in X_numeric.columns:
            if feature not in best_features:
                r = pearson_correlation(X_numeric[feature], y)
                if r > correlation:
                    correlation = r
                    best_feature = feature
        if best_feature is not None:
            best_features.append(best_feature)

    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        np.random.seed(self.random_state)
        self.theta = np.random.random(size=X.shape[1] + 1)
        self.thetas.append(self.theta)
        ones_column = np.ones(X.shape[0])
        X = np.c_[ones_column, X]
        for i in range(self.n_iter):
            sigmoid = 1 / (1 + np.exp(-np.dot(X, self.theta)))
            m = X.shape[0]
            J = (- 1 / m) * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
            self.theta = self.theta - self.eta * np.dot(sigmoid - y, X)
            self.Js.append(J)
            self.thetas.append(self.theta)
            if i > 0 and np.abs(self.Js[i] - self.Js[i - 1]) < self.eps:
                break



    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        preds = []

        ones_column = np.ones(X.shape[0])
        X = np.c_[ones_column, X]
        for i in range(X.shape[0]):
            sigmoid = 1 / (1 + np.exp(-np.dot(X[i], self.theta)))
            if sigmoid > 0.5:
                preds.append(1)
            else:
                preds.append(0)

        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    np.random.seed(random_state)
    accuracies = []
    random_indices = np.random.permutation(X.shape[0])
    indices_folds = np.array_split(random_indices, folds)
    for i in range(folds):
        training_arrays = indices_folds[:i] + indices_folds[i + 1:]
        train_indices = np.concatenate(training_arrays)
        validation_indices = indices_folds[i]
        X_train, X_validation = X[train_indices], X[validation_indices]
        y_train, y_validation = y[train_indices], y[validation_indices]
        algo.fit(X_train, y_train)
        preds = algo.predict(X_validation)
        accuracy = np.sum(preds == y_validation) / y_validation.size
        accuracies.append(accuracy)
    cv_accuracy = np.mean(accuracies)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    pi_part = 1 / (sigma * np.sqrt(2 * np.pi))
    exp_part = -0.5 * ((data - mu) / sigma) ** 2
    p = pi_part * np.exp(exp_part)

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.responsibilities = np.zeros((data.shape[0], self.k))
        self.weights = np.ones(self.k) / self.k
        splits = np.array_split(data, self.k)
        self.mus = np.empty(self.k)
        self.sigmas = np.empty(self.k)
        for i in range(self.k):
            self.mus[i] = np.mean(splits[i])
            self.sigmas[i] = np.std(splits[i])

        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        self.responsibilities = np.zeros((data.shape[0], self.k))
        data_one_dimensional = data.flatten()
        for i in range(data.shape[0]):
            for j in range(self.k):
                self.responsibilities[i][j] = self.weights[j] * norm_pdf(data_one_dimensional[i], self.mus[j], self.sigmas[j])
        self.responsibilities = self.responsibilities / np.sum(self.responsibilities, axis=1).reshape(-1, 1)



    def maximization(self, data):
        """
        M step - This function calculates and updates the model parameters
        """
        number_of_data_points = data.shape[0]
        self.weights = np.mean(self.responsibilities, axis=0).flatten()

        responsibilities_weighted_sum = np.dot(self.responsibilities.T, data)
        responsibilities_weighted_sum = responsibilities_weighted_sum.flatten()
        weights_mult_N = np.zeros(self.k)
        for j in range(self.k):
            weights_mult_N[j] = self.weights[j] * number_of_data_points
            self.mus[j] = responsibilities_weighted_sum[j] / weights_mult_N[j]

        self.sigmas = np.zeros(self.k)
        for j in range(self.k):
            self.sigmas[j] = np.sqrt(np.dot(self.responsibilities[:, j], (data.flatten() - self.mus[j]) ** 2) / weights_mult_N[j])


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization functions to estimate the distribution parameters.
        Store the parameters in the attributes of the EM object.
        Stop the function when the difference between the previous cost and the current cost is less than the specified epsilon
        or when the maximum number of iterations is reached.

        Parameters:
        - data: The input data for training the model.
        """
        self.init_params(data)
        for iteration in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            total_pdf = np.zeros_like(data)
            for j in range(self.k):
                total_pdf += self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
            log_likelihood = -np.sum(np.log(total_pdf))
            self.costs.append(log_likelihood)

            if iteration > 0 and np.abs(self.costs[-2] - self.costs[-1]) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    pdf = np.sum(weights * norm_pdf(data, mus, sigmas))

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.distributions = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        labels = np.unique(y)
        self.prior = {}
        self.distributions = {label: [None] * X.shape[1] for label in labels}
        for label in labels:
            label_data = X[y == label]
            self.prior[label] = label_data.shape[0] / X.shape[0]
            for feature in range(X.shape[1]):
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(label_data[:, feature])
                weights, mus, sigmas = em.get_dist_params()
                self.distributions[label][feature] = (weights, mus, sigmas)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        preds = []
        for instance in X:
            max_posterior = -1
            best_label_for_instance = None
            for label in self.prior:
                posterior = self.prior[label]
                for feature in range(X.shape[1]):
                    weights, mus, sigmas = self.distributions[label][feature]
                    posterior *= gmm_pdf(instance[feature], weights, mus, sigmas)
                if posterior > max_posterior:
                    max_posterior = posterior
                    best_label_for_instance = label
            preds.append(best_label_for_instance)

        return preds

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ['blue', 'red']
    cmap = ListedColormap(colors)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.array(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    lor_train_preds = logistic_regression.predict(x_train)
    lor_train_acc = np.sum(lor_train_preds == y_train) / y_train.size
    lor_test_preds = logistic_regression.predict(x_test)
    lor_test_acc = np.sum(lor_test_preds == y_test) / y_test.size

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    bayes_train_preds = naive_bayes.predict(x_train)
    bayes_train_acc = np.sum(bayes_train_preds == y_train) / y_train.size
    bayes_test_preds = naive_bayes.predict(x_test)
    bayes_test_acc = np.sum(bayes_test_preds == y_test) / y_test.size

    print(f"Logistic Regression - Training Accuracy: {lor_train_acc}")
    print(f"Logistic Regression - Test Accuracy: {lor_test_acc}")
    print(f"Naive Bayes - Training Accuracy: {bayes_train_acc}")
    print(f"Naive Bayes - Test Accuracy: {bayes_test_acc}")

    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=logistic_regression, title="Logistic Regression Decision Boundaries")
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=naive_bayes, title="Naive Bayes Decision Boundaries")

    plt.plot(range(len(logistic_regression.Js)), logistic_regression.Js)
    plt.title("Logistic Regression Cost Function")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    np.random.seed(42)

    # Dataset A: Class 0
    mean1_class0_dataset_a = [4, -6, 5]
    cov1_class0_dataset_a = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]
    samples_class0_part1_dataset_a = multivariate_normal.rvs(mean=mean1_class0_dataset_a, cov=cov1_class0_dataset_a,
                                                             size=100)

    mean2_class0_dataset_a = [-4, -3, -1]
    cov2_class0_dataset_a = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]
    samples_class0_part2_dataset_a = multivariate_normal.rvs(mean=mean2_class0_dataset_a, cov=cov2_class0_dataset_a,
                                                             size=100)

    samples_class0_dataset_a = np.vstack((samples_class0_part1_dataset_a, samples_class0_part2_dataset_a))

    # Dataset A: Class 1
    mean1_class1_dataset_a = [-4, -3, 5]
    cov1_class1_dataset_a = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]
    samples_class1_part1_dataset_a = multivariate_normal.rvs(mean=mean1_class1_dataset_a, cov=cov1_class1_dataset_a,
                                                             size=100)

    mean2_class1_dataset_a = [4, -6, -1]
    cov2_class1_dataset_a = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]
    samples_class1_part2_dataset_a = multivariate_normal.rvs(mean=mean2_class1_dataset_a, cov=cov2_class1_dataset_a,
                                                             size=100)

    samples_class1_dataset_a = np.vstack((samples_class1_part1_dataset_a, samples_class1_part2_dataset_a))

    X_dataset_a = np.vstack((samples_class0_dataset_a, samples_class1_dataset_a))
    y_dataset_a = np.hstack((np.zeros(samples_class0_dataset_a.shape[0]), np.ones(samples_class1_dataset_a.shape[0])))

    # Dataset B: Class 0
    mean1_class0_dataset_b = [2, 1, 1.5]
    cov1_class0_dataset_b = [[25, 0, 0], [0, 1, 0], [0, 0, 2.25]]
    samples_class0_part1_dataset_b = multivariate_normal.rvs(mean=mean1_class0_dataset_b, cov=cov1_class0_dataset_b,
                                                             size=100)

    mean2_class0_dataset_b = [3, 2, 3]
    cov2_class0_dataset_b = [[25, 0, 0], [0, 1, 0], [0, 0, 2.25]]
    samples_class0_part2_dataset_b = multivariate_normal.rvs(mean=mean2_class0_dataset_b, cov=cov2_class0_dataset_b,
                                                             size=100)

    samples_class0_dataset_b = np.vstack((samples_class0_part1_dataset_b, samples_class0_part2_dataset_b))

    # Dataset B: Class 1
    mean1_class1_dataset_b = [-1, -5, -7.5]
    cov1_class1_dataset_b = [[25, 0, 0], [0, 1, 0], [0, 0, 2.25]]
    samples_class1_part1_dataset_b = multivariate_normal.rvs(mean=mean1_class1_dataset_b, cov=cov1_class1_dataset_b,
                                                             size=100)

    mean2_class1_dataset_b = [-2, -6, -9]
    cov2_class1_dataset_b = [[25, 0, 0], [0, 1, 0], [0, 0, 2.25]]
    samples_class1_part2_dataset_b = multivariate_normal.rvs(mean=mean2_class1_dataset_b, cov=cov2_class1_dataset_b,
                                                             size=100)

    samples_class1_dataset_b = np.vstack((samples_class1_part1_dataset_b, samples_class1_part2_dataset_b))

    X_dataset_b = np.vstack((samples_class0_dataset_b, samples_class1_dataset_b))
    y_dataset_b = np.hstack((np.zeros(samples_class0_dataset_b.shape[0]), np.ones(samples_class1_dataset_b.shape[0])))

    dataset_a_features = (
    samples_class0_part1_dataset_a, samples_class0_part2_dataset_a, samples_class1_part1_dataset_a,
    samples_class1_part2_dataset_a)
    dataset_a_labels = (0, 0, 1, 1)
    dataset_b_features = (
    samples_class0_part1_dataset_b, samples_class0_part2_dataset_b, samples_class1_part1_dataset_b,
    samples_class1_part2_dataset_b)
    dataset_b_labels = (0, 0, 1, 1)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    plot_2D(X_dataset_a, y_dataset_a, X_dataset_b, y_dataset_b, axs)
    plot_3D(X_dataset_a, y_dataset_a, X_dataset_b, y_dataset_b)

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }


def plot_2D(X_dataset_a, y_dataset_a, X_dataset_b, y_dataset_b, axs):
    axs[0, 0].scatter(X_dataset_a[y_dataset_a == 0][:, 0], X_dataset_a[y_dataset_a == 0][:, 1], label='Class 0')
    axs[0, 0].scatter(X_dataset_a[y_dataset_a == 1][:, 0], X_dataset_a[y_dataset_a == 1][:, 1], label='Class 1')
    axs[0, 0].set_xlabel('Feature 1')
    axs[0, 0].set_ylabel('Feature 2')
    axs[0, 0].set_title('Dataset A: Feature 1 vs Feature 2')
    axs[0, 0].legend()

    axs[0, 1].scatter(X_dataset_a[y_dataset_a == 0][:, 0], X_dataset_a[y_dataset_a == 0][:, 2], label='Class 0')
    axs[0, 1].scatter(X_dataset_a[y_dataset_a == 1][:, 0], X_dataset_a[y_dataset_a == 1][:, 2], label='Class 1')
    axs[0, 1].set_xlabel('Feature 1')
    axs[0, 1].set_ylabel('Feature 3')
    axs[0, 1].set_title('Dataset A: Feature 1 vs Feature 3')
    axs[0, 1].legend()

    axs[0, 2].scatter(X_dataset_a[y_dataset_a == 0][:, 1], X_dataset_a[y_dataset_a == 0][:, 2], label='Class 0')
    axs[0, 2].scatter(X_dataset_a[y_dataset_a == 1][:, 1], X_dataset_a[y_dataset_a == 1][:, 2], label='Class 1')
    axs[0, 2].set_xlabel('Feature 2')
    axs[0, 2].set_ylabel('Feature 3')
    axs[0, 2].set_title('Dataset A: Feature 2 vs Feature 3')
    axs[0, 2].legend()

    axs[1, 0].scatter(X_dataset_b[y_dataset_b == 0][:, 0], X_dataset_b[y_dataset_b == 0][:, 1], label='Class 0')
    axs[1, 0].scatter(X_dataset_b[y_dataset_b == 1][:, 0], X_dataset_b[y_dataset_b == 1][:, 1], label='Class 1')
    axs[1, 0].set_xlabel('Feature 1')
    axs[1, 0].set_ylabel('Feature 2')
    axs[1, 0].set_title('Dataset B: Feature 1 vs Feature 2')
    axs[1, 0].legend()

    axs[1, 1].scatter(X_dataset_b[y_dataset_b == 0][:, 0], X_dataset_b[y_dataset_b == 0][:, 2], label='Class 0')
    axs[1, 1].scatter(X_dataset_b[y_dataset_b == 1][:, 0], X_dataset_b[y_dataset_b == 1][:, 2], label='Class 1')
    axs[1, 1].set_xlabel('Feature 1')
    axs[1, 1].set_ylabel('Feature 3')
    axs[1, 1].set_title('Dataset B: Feature 1 vs Feature 3')
    axs[1, 1].legend()

    axs[1, 2].scatter(X_dataset_b[y_dataset_b == 0][:, 1], X_dataset_b[y_dataset_b == 0][:, 2], label='Class 0')
    axs[1, 2].scatter(X_dataset_b[y_dataset_b == 1][:, 1], X_dataset_b[y_dataset_b == 1][:, 2], label='Class 1')
    axs[1, 2].set_xlabel('Feature 2')
    axs[1, 2].set_ylabel('Feature 3')
    axs[1, 2].set_title('Dataset B: Feature 2 vs Feature 3')
    axs[1, 2].legend()

    plt.show()


def plot_3D(X_dataset_a, y_dataset_a, X_dataset_b, y_dataset_b):
    fig = plt.figure(figsize=(18, 8))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X_dataset_a[y_dataset_a == 0][:, 0], X_dataset_a[y_dataset_a == 0][:, 1], X_dataset_a[y_dataset_a == 0][:, 2], label='Class 0')
    ax.scatter(X_dataset_a[y_dataset_a == 1][:, 0], X_dataset_a[y_dataset_a == 1][:, 1], X_dataset_a[y_dataset_a == 1][:, 2], label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Dataset A: 3D plot')
    ax.legend()

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(X_dataset_b[y_dataset_b == 0][:, 0], X_dataset_b[y_dataset_b == 0][:, 1], X_dataset_b[y_dataset_b == 0][:, 2], label='Class 0')
    ax.scatter(X_dataset_b[y_dataset_b == 1][:, 0], X_dataset_b[y_dataset_b == 1][:, 1], X_dataset_b[y_dataset_b == 1][:, 2], label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Dataset B: 3D plot')
    ax.legend()

    plt.show()