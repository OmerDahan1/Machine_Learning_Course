import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.11,
            (0, 1): 0.19,
            (1, 0): 0.19,
            (1, 1): 0.51
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.3,
            (1, 0): 0.2,
            (1, 1): 0.3
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.1,
            (0, 0, 1): 0.15,
            (0, 1, 0): 0.1,
            (0, 1, 1): 0.15,
            (1, 0, 0): 0.1,
            (1, 0, 1): 0.15,
            (1, 1, 0): 0.1,
            (1, 1, 1): 0.15,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X_Y = self.X_Y
        X_Y_dependent = False
        for x in self.X:
            for y in self.Y:
                if not np.isclose(self.X_Y[(x, y)] ,self.X[x] * self.Y[y]):
                    X_Y_dependent = True

        return X_Y_dependent

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X_Y_given_C_independent = True
        for x in self.X:
            for y in self.Y:
                for c in self.C:
                    if not np.isclose(self.X_Y_C[(x,y,c)] / self.C[c], (self.X_C[(x, c)] / self.C[c]) * (self.Y_C[(y, c)] / self.C[c])):
                        X_Y_given_C_independent = False

        return X_Y_given_C_independent

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    log_p = rate ** k * np.exp(-rate) / np.math.factorial(k)
    log_p = np.log(log_p)

    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    likelihoods = np.zeros(len(rates))
    for i, rate in enumerate(rates):
        for sample in samples:
            likelihoods[i] += poisson_log_pmf(sample, rate)
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    max_likelihood_index = np.argmax(likelihoods)
    rate = rates[max_likelihood_index]

    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    mean = np.mean(samples)

    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    pi_part = 1 / np.sqrt(2 * np.pi * std ** 2)
    exp_part = np.exp(-0.5 * ((x - mean) / std) ** 2)
    p = pi_part * exp_part
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = self.dataset[dataset[:, -1] == class_value]
        self.mean = np.mean(self.class_dataset[:, :-1], axis=0)
        self.std = np.std(self.class_dataset[:, :-1], axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = self.class_dataset.shape[0] / self.dataset.shape[0]

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = 1
        for feature in range(self.class_dataset[:,:-1].shape[1]):
            likelihood *= normal_pdf(x[feature], self.mean[feature], self.std[feature])

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        class_0_posterior = self.ccd0.get_instance_posterior(x)
        class_1_posterior = self.ccd1.get_instance_posterior(x)
        if class_0_posterior > class_1_posterior:
            pred = 0
        else:
            pred = 1

        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correctly_classified = 0
    for instance in test_set:
        if map_classifier.predict(instance) == instance[-1]:
            correctly_classified += 1

    acc = correctly_classified / test_set.shape[0]

    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    x_features = x[:-1]
    dimension = x_features.shape[0]
    pi_part = (2 * np.pi) ** (dimension / 2)
    cov_part = np.sqrt(np.linalg.det(cov))
    exp_part = -0.5 * np.dot(np.dot((x_features - mean).T, np.linalg.inv(cov)), x_features - mean)
    exp_part = np.exp(exp_part)
    pdf = (1 / (pi_part * cov_part)) * exp_part

    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = dataset[dataset[:,-1] == class_value]
        self.mean = np.mean(self.class_dataset[:, :-1], axis=0)
        self.cov_matrix = np.cov(self.class_dataset[:, :-1], rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = self.class_dataset.shape[0] / self.dataset.shape[0]

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov_matrix)

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        class_0_prior = self.ccd0.get_prior()
        class_1_prior = self.ccd1.get_prior()
        if class_0_prior > class_1_prior:
            pred = 0
        else:
            pred = 1

        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        class_0_likelihood = self.ccd0.get_instance_likelihood(x)
        class_1_likelihood = self.ccd1.get_instance_likelihood(x)
        if class_0_likelihood > class_1_likelihood:
            pred = 0
        else:
            pred = 1

        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = dataset[dataset[:, -1] == class_value]

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        prior = self.class_dataset.shape[0] / self.dataset.shape[0]
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = 1.0
        n_i = self.class_dataset.shape[0]
        v = []
        for j in range(self.dataset[:,:-1].shape[1]):
            v.append(len(np.unique(self.dataset[:, j])))

        for j in range(x[:-1].shape[0]):
            if x[j] not in self.dataset[:, j]:
                likelihood *= EPSILLON
            else:
                n_i_j = np.sum(self.class_dataset[:, j] == x[j])
                likelihood *= (n_i_j + 1) / (n_i + v[j])

        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        class_0_posterior = self.ccd0.get_instance_posterior(x)
        class_1_posterior = self.ccd1.get_instance_posterior(x)
        if class_0_posterior > class_1_posterior:
            pred = 0
        else:
            pred = 1

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correctly_classified = 0
        for instance in test_set:
            if self.predict(instance) == instance[-1]:
                correctly_classified += 1
        acc = correctly_classified / test_set.shape[0]

        return acc


