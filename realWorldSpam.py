import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import sqlite3
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import nltk
from nltk.tokenize import word_tokenize


#Load Amazon review Data Set using pandas
con = sqlite3.connect("D:\Artificial Intelligence\Real World Problem 1\database.sqlite")
df = pd.read_sql_query("Select * from Reviews where score != 3",con)


df["Score"] = [1 if x >=4 else 0 for x in df["Score"].values]
#Sorted the data set according to timestamp inorder to do the time base slicing
df  =  df.sort_values("Time")

#Remove The Dublicate
df  = df.drop_duplicates(["ProfileName","Time","UserId","Text"],keep='first')
df  = df[df["HelpfulnessNumerator"]<=df["HelpfulnessDenominator"]]
# Text Preprocessing:

words = set(stopwords.words('english'))
snb = SnowballStemmer('english')


def removeHTML(sentence):
    compiled = re.compile("<.*?>")
    newSentence = re.sub(compiled, '', sentence)
    return newSentence


def removePunct(sentence):
    compiled = re.compile("[!|[|.|,|/|)|(|^|`]")
    newSentence = re.sub(compiled, '', sentence)
    return newSentence


completeText = []
arrangeText = ''
df = df.iloc[:1000,:]

for sentence in df["Text"].values:
    word_ar = []
    sent = removeHTML(sentence)
    for word in removePunct(sent).split():
        if (len(word) >= 2):
            if (word.lower() not in words):
                word_ar.append(snb.stem(word.lower()))
            else:
                continue
        else:
            continue

    arrangeText = ' '.join(word_ar)
    completeText.append(arrangeText)

df["newText"] = completeText

X = np.array(df["newText"])
Y = np.array(df["Score"])

# now train_test_split data
x_tr,x_test,y_tr,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


#MSE error Misclassification error

def mseError(lis):
    return [1-x for x in lis]

# #Cross Validation function
def crossValidation(neighbourse,xTrain,yTrain):
    data  = []
    for i in neighbourse:
        knn = KNeighborsClassifier(n_neighbors=i, algorithm="kd_tree")
        scores = cross_val_score(knn, xTrain, yTrain, cv=10, scoring='accuracy')
        data.append(scores.mean())
    return data
#
# #find the optimal K
def OptimalKFinder(nei,mse):
    return nei[mse.index(min(mse))]

#calculating the test accuracy
def testAccuracy(xTrain,yTrain,optimalk,xTest,yTest):
    neigh = KNeighborsClassifier(optimalk)
    neigh.fit(xTrain, yTrain)
    y_pred = neigh.predict(xTest)
    print(accuracy_score(yTest, y_pred, normalize=True) * float(100))

#now we have train data now we can find the vector

#BAG OF WORDS

cv = CountVectorizer()
denseMat  = cv.fit_transform(x_tr)

svd =  TruncatedSVD(n_components=2)
x_tr_mat = svd.fit_transform(denseMat)


denseMattest = cv.fit_transform(x_test)
svd =  TruncatedSVD(n_components=2)
x_test_mat = svd.fit_transform(denseMattest)

neighbourse = np.arange(1,50,2)
bestK = crossValidation(neighbourse,x_tr_mat,y_tr)
MSEe = mseError(bestK)

# determining best k
optimal_k = OptimalKFinder(neighbourse,MSEe)
print('\nThe optimal number of neighbors is %d.' % optimal_k)

testAccuracy(x_tr_mat,y_tr,optimal_k,x_test_mat,y_test)

# Applying TFIDF

tdfid =  TfidfVectorizer()
denseTFIDmat = tdfid.fit_transform(x_tr)
svdTF = TruncatedSVD(n_components=2)
x_tfid_tr = svdTF.fit_transform(denseTFIDmat)


denseTFIDTestMat =tdfid.fit_transform(x_test)
svdTFtest = TruncatedSVD(n_components=2)
x_tfid_test = svdTFtest.fit_transform(denseTFIDTestMat)


neigbourTFID = np.arange(1,50,2)
scoringListTFID = crossValidation(neigbourTFID,x_tfid_tr,y_tr)
MSE = mseError(scoringListTFID)

optimal_k_tfid = OptimalKFinder(neigbourTFID,MSE)
print('\nThe optimal number of neighbors is %d.' % optimal_k_tfid)
testAccuracy(x_tfid_tr,y_tr,optimal_k_tfid,x_tfid_test,y_test)

#Word to vector using the gensim


def tokenizeData(data):
    list_of_sentence = []
    for sent in data:
        filtered_sentence = []
        for word in sent.split():
            filtered_sentence.append(word)

        list_of_sentence.append(filtered_sentence)

    return list_of_sentence

def getw2v(tokenizedData):
    x_tr_data = []
    for sent in tokenizedData:
        sent_vec = np.zeros(50)
        for word in sent:
            try:
                sent_vec += w2v.wv[word]
            except:
                pass

        x_tr_data.append(sent_vec)
    return x_tr_data


tokizedData_tr_data = tokenizeData(x_tr)
tokizedData_test_data =  tokenizeData(x_test)

w2v = gensim.models.Word2Vec(tokizedData_tr_data,min_count=5,size=50,workers=4)

x_tr_vector = getw2v(tokizedData_tr_data)
x_test_vector  = getw2v(tokizedData_test_data)

svd_w2v_tr = TruncatedSVD(n_components=2)
x_tr_svd = svd_w2v_tr.fit_transform(x_tr_vector)

svd_w2v_test = TruncatedSVD(n_components=2)
x_test_svd = svd_w2v_test.fit_transform(x_test_vector)

neig = np.arange(1,50,2)
data = crossValidation(neig,x_tr_svd,y_tr)

MSE_Err = mseError(data)
#Optimal K
optimalKVal = OptimalKFinder(neig,MSE_Err)
print('\nThe optimal number of neighbors is %d.' % optimalKVal)
testAccuracy(x_tr_svd,y_tr,optimalKVal,x_test_svd,y_test)


#Average Word To vector :


def avgW2V(data):
    avgw2v =[]
    for sent in data:
        sent_lis = np.zeros(50)
        cnt =0
        for word in sent:
            try:
                sent_lis += w2v.wv[word]
                cnt +=1
            except:
              pass

        sent_lis /= cnt
        avgw2v.append(sent_lis)
    return avgw2v

x_avgw2v_tr = avgW2V(tokizedData_tr_data)
x_avgw2v_test = avgW2V(tokizedData_test_data)


avg_w2v_data= crossValidation(neig,x_avgw2v_tr,y_tr)
MSE_Err_avg_w2v = mseError(avg_w2v_data)
optimalKVal_w2v = OptimalKFinder(neig,MSE_Err_avg_w2v)
print('\nThe optimal number of neighbors is %d.' % optimalKVal_w2v)
testAccuracy(x_avgw2v_tr,y_tr,optimalKVal_w2v,x_avgw2v_test,y_test)


#Average TFID

features  = tdfid.get_feature_names()

def avgTFIDF(data):
    row = 0
    avg_tfid =[]
    for sent in data:
        total_TFIDF = 0
        w2v_lis = np.zeros(50)
        for word in sent:
           try:

               wv_data = w2v.wv[word]
               tfid = denseTFIDmat[row,features.index(word)]
               newW2v = (wv_data*tfid)
               total_TFIDF +=1
               w2v_lis += newW2v
           except:
               pass


        w2v_lis /= total_TFIDF
        avg_tfid.append(w2v_lis)

    return avg_tfid

x_avgtfid_tr = avgTFIDF(tokizedData_tr_data)
x_avgtfid_test=avgTFIDF(tokizedData_test_data)


avg_avgtfid_data= crossValidation(neig,x_avgtfid_tr,y_tr)
MSE_Err_avg_tfid = mseError(avg_avgtfid_data)
optimalKVal_avgTFID = OptimalKFinder(neig,MSE_Err_avg_tfid)
print('\nThe optimal number of neighbors is %d.' % optimalKVal_avgTFID)
testAccuracy(x_avgtfid_tr,y_tr,optimalKVal_avgTFID,x_avgtfid_test,y_test)


