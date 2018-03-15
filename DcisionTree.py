from sklearn.feature_extraction import  DictVectorizer   #decision tree
import csv
from sklearn import (tree,preprocessing)
from sklearn.externals.six import StringIO

allElectronicsData=open(r'C:/Users/Administrator/PycharmProjects/test/training_code/AllElectronics.csv','rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)

featureList=[]
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict={}
    for i in range(len(row)-1):
        rowDict[headers[i+1]]=row[i+1]
    featureList.append(rowDict)
# print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print('dummyX:\n'+str(dummyX))
print(vec.get_feature_names())
print("labelList: " + str(labelList))
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf:\n"+str(clf))

with open('allElectronicInformationGainOri.dot','w') as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
