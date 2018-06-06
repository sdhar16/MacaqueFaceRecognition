import csv,sys,pickle,numpy,random,os,time,cv2,pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.feature import local_binary_pattern as lbp
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
def Face_Train(datafile):
    dataset = open(datafile,"rb")
    X_train,y_train,X_test,y_test = pickle.load(dataset)
    dataset.close()
    print(X_train.shape,X_test.shape,y_test.shape)
    c = 0.001
    cval=[]
    acc = []
    while(c<=10000):
        model = SVC(kernel="linear",C=c)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,pred)*100
        cval.append(c)
        acc.append(accuracy)
        print(c,accuracy)
        c *= 10
    dataout = open("values.pkl","wb")
    pickle.dump([cval,acc],dataout)
    dataout.close()
    
    

if __name__ == '__main__':
    Face_Train("dataMacaque.pkl")    
