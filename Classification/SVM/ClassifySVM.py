from sklearn.svm import SVC


def classify(features_train, labels_train):   
    ### import the sklearn module for SVM
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    

    return clf
