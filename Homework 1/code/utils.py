import os
import cv2
import numpy as np
import scipy.spatial
from timeit import default_timer as timer
from sklearn import neighbors, svm, cluster, preprocessing
from random import sample

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels



def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    if is_linear:
        svc = svm.LinearSVC(C=svm_lambda)
        svc.fit(train_features, train_labels)
        return svc.predict(test_features)
    else:
        classifiers = []
        for i in range(15):
            arr = train_labels[:]
            for j in range(len(train_labels)):
                if arr[j] != i:
                    arr[j] = -1
            svc = svm.SVC(kernel="rbf",C=svm_lambda ,gamma='scale',probability=True)
            svc.fit(list(train_features),arr)
            classifiers.append(svc)
        probs = []
        for i,cf in enumerate(classifiers):
            probabilities = cf.predict_proba(test_features).tolist()
            probs.append([prob[1] for prob in probabilities])
        probs = np.transpose(np.array(probs))
        return np.argmax(probs, axis=1).tolist()


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    resized_image = cv2.resize(input_image, (target_size, target_size))
    output_image = np.zeros(shape=resized_image.shape)
    output_image = cv2.normalize(src=resized_image, dst=output_image, alpha=-1, beta=1, dtype=cv2.CV_32F)
    output_image = output_image.flatten()
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    count = 0 
    for i in range(len(true_labels)):
        if predicted_labels[i] == true_labels[i]:
            count += 1
    accuracy = count / len(predicted_labels) * 100
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image
    # features = []
    # if feature_type == 'sift':
    #     sift = cv2.xfeatures2d.SIFT_create(nfeatures=25)
    #     vocabulary = []
    #     for i in train_images:
    #         k, d = sift.detectAndCompute(i, None)
    #         for des in d:
    #             features.append(des)
    #     #print(np.shape(features))
    #     if clustering_type == 'kmeans':
    #         clus = cluster.KMeans(n_clusters=dict_size).fit(features)
    #         vocabulary = clus.cluster_centers_
    #     elif clustering_type == 'hierarchical':
    #         clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
    #         dictionary = []
    #         for i in range(clus.n_clusters):
    #             dictionary[i] = []
    #         for i in range(len(features)):
    #             dictionary[clus.labels_[i]] = dictionary[clus.labels_[i]].append(features[i])
    #         for i in dictionary:
    #             #print(np.shape(dictionary[i]))
    #             arr = []
    #             tmp = np.array(dictionary[i])
    #             for j in range(len(dictionary[i][0])):
    #                 arr.append(np.mean(tmp[:,j]))
    #             vocabulary.append(arr) 
            
    # elif feature_type == 'surf':
    #     surf = cv2.xfeatures2d.SURF_create()
    #     features = []
    #     vocabulary = []
    #     for i in train_images:
    #         k, d = surf.detectAndCompute(i, None)
    #         for des in d:
    #             features.append(des)
    #     r = sample(features, 20)
    #     #print(np.shape(r))
    #     if clustering_type == 'kmeans':
    #         print('fitting kmeans surf')
    #         clus = cluster.KMeans(n_clusters=dict_size).fit(r)
    #         vocabulary = clus.cluster_centers_
    #     elif clustering_type == 'hierarchical':
    #         #r = sample(features, len(features) * 0.1)
    #         print('fitting agglomerative surf')
    #         clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(r)
    #         dictionary = {}
    #         for i in range(dict_size):
    #             dictionary[i] = []
    #         for i in range(len(clus.labels_)):
    #             dictionary[clus.labels_[i]].append(r[i])
    #         for i in dictionary:
    #             #print(np.shape(dictionary[i]))
    #             arr = []
    #             tmp = np.array(dictionary[i])
    #             for j in range(len(dictionary[i][0])):
    #                 arr.append(np.mean(tmp[:,j]))
    #             vocabulary.append(arr)  
    #         #print(clus.labels_)
    # elif feature_type == 'orb':
    #     orb = cv2.ORB_create(nfeatures=25)
    #     features = []
    #     vocabulary = []
    #     for i in train_images:
    #         k = orb.detect(i, None)
    #         k, d = orb.compute(i, k)
    #         if d is None:
    #           continue
    #         for des in d:
    #             features.append(des)
    #     if clustering_type == 'kmeans':
    #         clus = cluster.KMeans(n_clusters=dict_size).fit(features)
    #         vocabulary = clus.cluster_centers_
    #     elif clustering_type == 'hierarchical':
    #         clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
    #         dictionary = {}
    #         for i in range(dict_size):
    #             dictionary[i] = []
    #         for i in range(len(clus.labels_)):
    #             dictionary[clus.labels_[i]].append(features[i])
    #         for i in dictionary:
    #             #print(np.shape(dictionary[i]))
    #             arr = []
    #             tmp = np.array(dictionary[i])
    #             for j in range(len(dictionary[i][0])):
    #                 arr.append(np.mean(tmp[:,j]))
    #             vocabulary.append(arr)             

    # return vocabulary
    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        features = []
        vocabulary = []
        for i in train_images:
            k, d = sift.detectAndCompute(i, None)
            for des in d:
                features.append(des)
        #print(np.shape(features))
        if clustering_type == 'kmeans':
            clus = cluster.KMeans(n_clusters=dict_size).fit(features)
            vocabulary = clus.cluster_centers_
        elif clustering_type == 'hierarchical':
            clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
            dictionary = {}
            for i in range(dict_size):
                dictionary[i] = []
            for i in range(len(clus.labels_)):
                dictionary[clus.labels_[i]].append(features[i])
            for i in dictionary:
                #print(np.shape(dictionary[i]))
                arr = []
                tmp = np.array(dictionary[i])
                for j in range(len(dictionary[i][0])):
                    arr.append(np.mean(tmp[:,j]))
                vocabulary.append(arr) 
            
    elif feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        features = []
        vocabulary = []
        for i in train_images:
            k, d = surf.detectAndCompute(i, None)
            r = sample(list(d), 20)
            for des in r:
                features.append(des)
         #   print(np.shape(features))
        #print(np.shape(r))
        if clustering_type == 'kmeans':
         #   print('fitting kmeans')
            clus = cluster.KMeans(n_clusters=dict_size).fit(features)
            vocabulary = clus.cluster_centers_
        elif clustering_type == 'hierarchical':
            #r = sample(features, len(features) * 0.1)
         #   print('fitting agglomerative')
            clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
            dictionary = {}
            for i in range(dict_size):
                dictionary[i] = []
            for i in range(len(clus.labels_)):
                dictionary[clus.labels_[i]].append(features[i])
            for i in dictionary:
                #print(np.shape(dictionary[i]))
                arr = []
                tmp = np.array(dictionary[i])
                for j in range(len(dictionary[i][0])):
                    arr.append(np.mean(tmp[:,j]))
                vocabulary.append(arr)  
            #print(clus.labels_)
    elif feature_type == 'orb':
        orb = cv2.ORB_create(nfeatures=25)
        features = []
        vocabulary = []
        for i in train_images:
            k = orb.detect(i, None)
            k, d = orb.compute(i, k)
            if d is None:
              continue
            for des in d:
                features.append(des)
        if clustering_type == 'kmeans':
            clus = cluster.KMeans(n_clusters=dict_size).fit(features)
            vocabulary = clus.cluster_centers_
        elif clustering_type == 'hierarchical':
            clus = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
            dictionary = {}
            for i in range(dict_size):
                dictionary[i] = []
            for i in range(len(clus.labels_)):
                dictionary[clus.labels_[i]].append(features[i])
            for i in dictionary:
                #print(np.shape(dictionary[i]))
                arr = []
                tmp = np.array(dictionary[i])
                for j in range(len(dictionary[i][0])):
                    arr.append(np.mean(tmp[:,j]))
                vocabulary.append(arr)             

    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    features = []
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        k, d = sift.detectAndCompute(image, None)
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        k, d = surf.detectAndCompute(image, None)
    else:
        orb = cv2.ORB_create()
        k = orb.detect(image, None)
        k, d = orb.compute(image, k)
        if d is None:
            return [value/len(vocabulary) for value in np.histogram([],bins=np.arange(len(vocabulary)))[0]]
    temp = scipy.spatial.distance.cdist(XA=d,XB=vocabulary,metric='euclidean')
    closest = np.argmin(temp, axis=1)
    Bow = np.histogram(closest,bins=np.arange(len(vocabulary)))

    return [value/len(vocabulary) for value in Bow[0]]


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier


    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    sizes = { 8, 16, 32 }
    neighbors = { 1, 3, 6}
    
    classResult = []
    total_time = 0.0
    for s in sizes:
        train_resized = []
        test_resized = []
        for image in train_features:
            train_resized.append(imresize(image, s))
        for image in test_features:
            test_resized.append(imresize(image, s))
        for n in neighbors:
            start = timer()
            prediction = KNN_classifier(train_resized, train_labels, test_resized, n)
            end = timer()
            total_time = end-start
            classResult.append(reportAccuracy(test_labels, prediction))
            classResult.append(total_time)
    return classResult
    