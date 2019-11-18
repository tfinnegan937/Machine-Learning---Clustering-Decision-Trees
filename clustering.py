import numpy as np


def get_centers(X, C, K):
    last_distance = distance(C[0], X[0]) #Storage for the distance of the last cluster checked
    new_c = np.zeros((K, X[0].size)) #output center
    closest_cluster = 0 #index of the cluster closest to the sample
    num_per_cluster = np.zeros(K) #Number of samples in each cluster, as a 1D array

    '''Iterate over all of the clusters. Find the closest cluster. Increase the
        number of samples in that cluster, and then add the sample to the cluster'''
    for i in range(0, X[...,0].size):
        for j in range(0, K):
            if last_distance > distance(C[j], X[i]):
                closest_cluster = j
                last_distance = distance(C[j], X[i])
        new_c[closest_cluster] = new_c[closest_cluster] + X[i]
        num_per_cluster[closest_cluster] = num_per_cluster[closest_cluster] + 1
    #Take each summed cluster, and then divide it by the number of samples in that cluster
    for cluster in range(0, K):
        if not(num_per_cluster[cluster] == 0):
            new_c[cluster] = new_c[cluster] / num_per_cluster[cluster]

    return new_c


def distance(P1, P2):
    #Takes the 2 norm
    return np.linalg.norm(P1-P2)


def K_Means_recur(X, C, K, depth):
    #Get the centers
    new_c = get_centers(X, C, K)
    cont = False
    #if the centers are no longer moving, don't continue. Else, continue
    #Caps out at a recursive depth of 600, otherwise it hits python's recursion limit
    for cluster in range(0, K):
        while np.array_equal(new_c[cluster], np.zeros(2)):
            new_c[cluster] = np.random.random_integers(np.amin(X), np.amax(X), X[0].size)
    for i in range(0, K):

        if not(distance(C[i], new_c[i]) == 0) and not(depth >= 100):
            cont = True
    if not cont:
        return new_c
    return K_Means_recur(X, new_c, K, depth + 1)


def K_Means(X, K):
    #randomly generate centers
    c_out = np.zeros((K, X[0].size)) #initialize output cluster
    C = np.zeros((K, X[0].size)) #initialize clusters to zero

    for cluster in range(0, K):
        C[cluster] = np.random.random_integers(np.amin(X), np.amax(X), X[0].size) #randomize starting clusters along max and min
    c_out = K_Means_recur(X, C, K, 0) # recursively generate the cluster

    while np.unique(c_out, axis=0)[..., 0].size != K: #if it contains duplicate clusters, regenerate
        c_out = K_Means(X, K)

    return c_out


def K_Means_better(X, K): #inefficient. Please wait. It will output
    #dictionary to hold the number of times a cluster appears
    num_dict = dict()
    #dictionary to hold the percentage of total times the cluster appears
    percent_dict = dict()

    percent_largest = 0.0
    times_performed = 1
    cluster = np.zeros((K, X[0].size))

    while (percent_largest < .50 or times_performed < 50) and times_performed < 100:
        #Lists do not work as keys. Convert to string to use as key
        new_mean = np.array2string(K_Means(X, K))
        #Add or increment cluster centers to dictionary
        if (new_mean in num_dict):
            num_dict[new_mean] = num_dict[new_mean] + 1
        else:
            num_dict[new_mean] = 1
        #Update all percentages
        for state in num_dict:
            percent_dict[state] = num_dict[state] / times_performed

        cluster_string = max(num_dict, key=num_dict.get)
        #Lines 70-75 convert the string back into an np array
        cluster_string_array = cluster_string[1:len(cluster_string) - 1].splitlines()

        for i in range(0, K):
            stre = cluster_string_array[i][cluster_string_array[i].index('[') + 1:cluster_string_array[i].index(']')]
            array = stre.split()
            for j in range(0, len(array)):
                cluster[i][j] = float(array[j])
        percent_largest = percent_dict[new_mean]
        times_performed = times_performed + 1

    return cluster
