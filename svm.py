import apriori
dataSet=apriori.loadDataSet()
L,suppData=apriori.apriori(dataSet,minSupport=0.5)
rules=apriori.generateRules(L,suppData,minConf=0.5)

from sklearn import svm

x=[[2,0],[1,1],[2,3]]
y = [0,0,1]
clf = svm.SVC(kernel='linear')
clf.fit(x,y)
print(clf)
print(clf.predict([[2,0]])) #predict[2,0] belong to which class


import numpy as np
import pylab as pl
from sklearn import svm
np.random.seed(0)
x = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
y=[0]*20+[1]*20  #[0,0...1,1]
clf = svm.SVC(kernel='linear')
clf.fit(x,y)
w=clf._get_coef()[0]
a=-w[0]/w[1]
xx=np.linspace(-5,5)
yy=a*xx-(clf._intercept_[0])/w[1]
b=clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b=clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print('w:'+ str(w))
print('a:'+ str(a))
print("support_vectors_:"+str(clf.support_vectors_))
print("clf.coef_:"+str(clf._get_coef()))
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
           s=80,facecolors='red')
pl.scatter(x[:,0],x[:,1],c=y,cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()

#Face recognition
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA, PCA  #decomposition:fenjie
from sklearn.svm import SVC
# display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
#download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
#introspect the images arrays to find the shapes (for plotting)
n_samples, h, w=lfw_people.images.shape
# for machine learning we use the 2 data directly （as relative pixel)
x= lfw_people.data
n_features = x.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_calsses = target_names.shape[0]
print('total dataset size:')
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_calsses)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.25)
n_components = 150
print("Extracting the top %d eigenfaces from %d faces"
       % (n_components,x_train.shape[0]))
t0=time()
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(x_train)
print("done in %0.3fs" % (time()-t0))

eigenfaces = pca.components_.reshape((n_components,h,w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("done in %0.3fs" % (time()-t0))

print("Fitting the classifier to the training set")
t0=time()
param_grid={'C':[1e3,5e3,1e4,5e4,1e5],
            'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
clf=clf.fit(x_train_pca,y_train)
print("done in %0.3fs" % (time()-t0))
print("Best estimator found by grid search:")   #网格搜索
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0=time()
y_pred = clf.predict(x_test_pca)
print("done in %0.3fs" % (time()-t0))

print(classification_report(y_test,y_pred,target_names=target_names))
print(confusion_matrix(y_test,y_pred,labels=range(n_calsses)))

def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8* n_col,2.4 * n_row))
    plt.subplots_adjust(bottom=0.03,left=0.01,right=0.99,top=0.9,hspace=0.35)
    for i in range(n_row *n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    return 'predicted: %s \ntrue:       %s' % (pred_name,true_name)

prediction_titles = [title(y_pred,y_test,target_names,i)
                     for i in range(y_pred.shape[0])]
plot_gallery(x_test,prediction_titles,h,w)

eigenfaces_titles = ['eigenface %d' % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenfaces_titles,h,w)
plt.show()