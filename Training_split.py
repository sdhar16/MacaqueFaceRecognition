import os
import pickle
import random

import numpy
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

import cv2


def faceTrain(image_dir,output_file,cval):
    data = [] #Contains file names for image location
    for root, _, filenames in os.walk(image_dir):
        for filename in filenames:
            data.append(os.path.join(root,filename)) #Adding file names
    idList = {'Dan': 1, 'Judd': 2, 'Lala': 3, 'Leah': 4, 'Libby': 5, 'Linz': 6, 'Love': 7, 'Lydia': 8, 'Maj': 9, 'Meesha': 10,
     'Meg': 11, 'Melody': 12, 'Mindy': 13, 'Ocelot': 14, 'Rupee': 15, 'Saphy': 16,
     'Serena': 17, 'Shirley': 18, 'Sizzle': 19, 'Sol': 20, 'Sonja': 21, 'Spice': 22, 'Star': 23, 'Sugar': 24, 'Tamara': 25, 
     'Tass': 26, 'Tea': 27, 'Teal': 28, 'Tes': 29, 'Thyme': 30, 'Umbrella': 31, 'Ursula': 32, 'Venus': 33, 'Verity': 34}
    #Idlist is the name of every monkey assigned a unique natural number
    imageData = [] #Will contain the feature vector of each image data
    output = []  # Is the output of the data a.k.a TARGET. Will contain a number corresponding to each unique monkey from IDLIST 
    no_of_pixels = 58 #no of pixels to be considered during LBP of each cell of the image
    #random.shuffle(data) #Shuffling the data
    testing_for = "Rotated"
    for image_location in data :
        #print(image_location.split("/")[-3])
        if(image_location.split("/")[-3]==testing_for):
            continue
        image = cv2.imread(image_location,0) #Read the image form the file in GRAYSCALE mode
        window_size = 20 #Cell side has been taken to be 20x20.
        #print(lbp.shape)
        lbp = local_binary_pattern(image,8,1,method="nri_uniform") #Finding LBP of image with neighbors=8 and radius=1 with method uniform non rotation
        imageHist = [] # Contains the extravted LBP feature vector
        for r in range(0,image.shape[0]-19,window_size):
            for c in range(0,image.shape[1]-19,window_size):
                window = lbp[r:r+window_size,c:c+window_size] #Calculate the window
                hist,_ = numpy.histogram(window.flatten(),no_of_pixels + 1,[0,no_of_pixels]) #Find the histrogram of the image
                imageHist.extend(hist) #Put the extracted set of features from this window to the feature vector list
        #print(len(imageHist))
        imageData.append(imageHist) #Add feature vector of the image
        output.append(idList[image_location.split("/")[-2]]) #Add target number for the monkey
        #print(count)
    test_data=[]
    test_output = []
    for image_location in data :
        if(image_location.split("/")[-3]!=testing_for):
            continue
        image = cv2.imread(image_location,0) #Read the image form the file in GRAYSCALE mode
        window_size = 20 #Cell side has been taken to be 20x20.
        #print(lbp.shape)
        lbp = local_binary_pattern(image,8,1,method="nri_uniform") #Finding LBP of image with neighbors=8 and radius=1 with method uniform non rotation
        imageHist = [] # Contains the extravted LBP feature vector
        for r in range(0,image.shape[0]-19,window_size):
            for c in range(0,image.shape[1]-19,window_size):
                window = lbp[r:r+window_size,c:c+window_size] #Calculate the window
                hist,_ = numpy.histogram(window.flatten(),no_of_pixels + 1,[0,no_of_pixels]) #Find the histrogram of the image
                imageHist.extend(hist) #Put the extracted set of features from this window to the feature vector list
        #print(len(imageHist))
        test_data.append(imageHist) #Add feature vector of the image
        test_output.append(idList[image_location.split("/")[-2]]) #Add target number for the monkey
        #print(count)
    imageData = numpy.array(imageData) #Convert to numpy array
    output = numpy.array(output) #Convert to numpy array
    test_data = numpy.array(test_data)
    test_output = numpy.array(test_output)
    # imageData = normalize(imageData,norm='l2')
    # test_data = normalize(test_data,norm='l2')
    pca = PCA(0.96)
    pca.fit(imageData)
    print("Number of components",pca.n_components_) #Find the new dimensions
    imageDatapca = pca.transform(imageData)
    test_datapca = pca.transform(test_data)
    model = SVC(kernel="linear",C=1000)
    model.fit(imageDatapca,output)
    pred = model.predict(test_datapca)
    print("Accuracy SVM with testData=",testing_for,accuracy_score(test_output,pred)*100) #Find the accuracy
    clf = LDA() #Set LDA model 
    print("Training LDA") #Printing training of LDA
    clf.fit(imageData,output) #Fit the model 
    pred1 = clf.predict(test_data) #Predict the output for test file
    print("Accuracy LDA for testdata=",testing_for,accuracy_score(test_output,pred1)*100) #Find the accuracy
    '''X_train,X_test,y_train,y_test = train_test_split(imageData,output,test_size = 0.33) #Split data to train and test with ratio 6.7:3.3. 
    #Not advised in the paper 
    elements,count = numpy.unique(y_train,return_counts=True)
    pca = PCA(0.96) #reduced dimensions to 96% variance
    pca.fit(X_train) #Fit the training data
    print("Number of components",pca.n_components_) #Find the new dimensions
    X_train = pca.transform(X_train) #Transform the feature matrix
    X_test = pca.transform(X_test) #Transform the feature matrix
    print("Data Processed")
    model = SVC(kernel = "linear",C=100) #Set model for SVM
    model.fit(X_train,y_train) #Train the SVM with the data
    print("trained")
    pred = model.predict(X_test) #Predict the test data
    print("Accuracy",accuracy_score(y_test,pred)*100) #Find the accuracy
    #########################################################################
    clf = LDA() #Set LDA model 
    print("Training LDA") #Printing training of LDA
    clf.fit(X_train,y_train) #Fit the model 
    pred1 = clf.predict(X_test) #Predict the output for test file
    print("Accuracy",accuracy_score(y_test,pred1)*100) #Find the accuracy
    '''


if __name__ == '__main__':
    faceTrain("MacaqueFaces/","cost",10) #MacaqueFaces contains the data to monkeys arranged properly. please see the directory structure
