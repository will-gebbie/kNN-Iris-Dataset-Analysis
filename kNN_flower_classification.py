import numpy as np
import pandas as pd
from sklearn import neighbors
# Used to split data set into training and test data
from sklearn.model_selection import train_test_split


# Initialize Iris Dataset as a Pandas Dataframe
colnames = ['sep_length', 'sep_width', 'pet_length', 'pet_width', 'label']
flower_features = pd.read_csv('iris.data', names=colnames)

# Initialize Data
inputMatrix = flower_features[colnames[:4]]
outputVector = flower_features['label']

# Split Training Data and Test Data as 70/30 ratio from the original dataset
X_train, X_test, y_train, y_test = train_test_split(
    inputMatrix, outputVector, test_size=.30, random_state=943)

# List of accuracies to be pulled from each iteration
accuracies = []
k_values = []

for k in range(y_train.shape[0]):
    # Initialize KNN Classfier with k=7
    clf = neighbors.KNeighborsClassifier(n_neighbors=k+1)

    k_values.append(k+1)  # start at k=1
    '''
    KNN fitting method fhat(x) = Ave(yi|xi ∈ Nk(x))

    Given an observation xi (a single 4 dim input) take the average
    of the k nearest yi's (labels: setosa, versicolour, or virginica)
    in the neighborhood Nk(x = xi) and assign that neighborhood as a zone for
    classifying future points as that average of the labels.

    As k increases to be an integer equivalent to an increasing percentage of
    the training data, the the neighborhood sizes shift from a local scale to
    a more global scale. This should cause the average of those neigborhoods to
    be less accurate given the specific observation in the training data xi.
    '''
    clf.fit(X_train, y_train)

    '''
    Since the input is only 4 dimensional, kNN should work decently. But as the
    dimensionality of the input matrix increases, neighborhoods become hypercubes
    of their current input space on a global scale rather than a local scale.
    '''

    # Cross Validation
    # the score function compares the prediction it makes on the X_test data and the y_test actual value to compute accuracy of the classifer
    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)
    num_correct = accuracy * y_test.shape[0]

    """ 
    print("Number of Neighbors = {}".format(k+1))
    print("Correctly Predicted {}/{} test observations!".format(int(num_correct),
                                                                y_test.shape[0]))
    print("Accuracy = {0:.4f}%".format(accuracy*100))
    print() 
    """

# Create a dataframe (matrix) of columns k and Accuracy
k_vs_accuracy_matrix = np.column_stack((k_values, accuracies))
k_vs_acc_df = pd.DataFrame(k_vs_accuracy_matrix, columns=['k', 'Accuracy'])

# correlation dataframe of k and Accuracy which returns the correlation coefficients relative to each variable
correlation_df = k_vs_acc_df.corr()
print("Correlation Matrix: ")
print('------------------------------')
print(correlation_df)
print('------------------------------')
print("As k increases, accuracy decreases")
print()

print("10 Most Accurate k-Values for predicting the Iris Dataset (given random_state = 943): ")
# returns highest to lowest order of datapoints based on the accuracy column
print(k_vs_acc_df.nlargest(10, columns='Accuracy'))
