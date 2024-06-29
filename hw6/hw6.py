import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]

    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64)

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    for i in range(k):
        distances.append(np.linalg.norm(X - centroids[i], ord=p, axis=1))
    distances = np.array(distances)

    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        min_centroid_index = np.argmin(distances, axis=0)
        new_centroids = np.empty((k, 3))
        for j in range(k):
            new_centroids[j] = np.mean(X[min_centroid_index == j], axis=0)

        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    classes = min_centroid_index

    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None

    centroids = get_random_centroids(X, 1)
    X_remaining_instances = np.delete(X, np.where(np.all(X == centroids[0], axis=1)), axis=0)
    centroids = np.array(centroids)

    while (centroids.shape[0] < k):
        distances = lp_distance(X_remaining_instances, centroids, p)
        min_distances = np.min(distances, axis=0)
        squared_distances = min_distances ** 2
        probabilities = squared_distances / np.sum(squared_distances)
        new_centroid_index = np.random.choice(X_remaining_instances.shape[0],1, p=probabilities)
        centroids = np.vstack((centroids, X_remaining_instances[new_centroid_index]))
        X_remaining_instances = np.delete(X_remaining_instances, np.where(np.all(X_remaining_instances == X_remaining_instances[new_centroid_index], axis=1)), axis=0)
    centroids = np.array(centroids)

    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        min_centroid = np.argmin(distances, axis=0)
        new_centroids = np.empty((k, 3))
        for j in range(k):
            new_centroids[j] = np.mean(X[min_centroid == j], axis=0)

        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    classes = min_centroid

    return centroids, classes
