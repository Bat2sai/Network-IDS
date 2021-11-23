#!/usr/bin/env python
# coding: utf-8

# <p style="font-size:36px;text-align:center"> <b>Network Intrusion Detection System </b> </p>
# <p style="font-size:36px;text-align:center"> <b> (NIDS - KDD CUP 99) </b> </p>

# .

# <h1>1 :  Business/Real-world Problem</h1>

# ## 1.1 : what is Nerwork intrusion ?

# A <b>network intrusion</b> is any unauthorized activity on a computer network.
# The  unauthorized activities or abnormal network activities threaten users' privacy and potentially damage the function and infrastructure of the whole network
#
# ##### Intrusion detector :
# The Intrusion detection system will detect network intrusions protects a computer network from unauthorized users, including perhaps insiders

# ## 1.2 : Problem Statement

#  The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between bad connections, called intrusions or attacks, and good normal connections.

# <h2>1.3 : Source/Useful Links </h2>

# The data set used here is NSL KDD (new version of kdd-cup99)
#
# The 1998 DARPA Intrusion Detection Evaluation Program was prepared and managed by MIT Lincoln Labs. The objective was to survey and evaluate research in intrusion detection.  A standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment, was provided.  The 1999 KDD intrusion detection contest uses a version of this dataset.
#
# NSL_KDD which is the new version of kdd-cup99 has the following advantages:
#
# - No redundant records in the train set, so the classifier will not produce any biased result
# -  No duplicate record in the test set which have better reduction rates.
# - The number of selected records from each difficult level group is inversely proportional to the percentage of records in the original KDD data set  
#
#
#
# In this dataset Attacks fall into four main categories:
#
# * DOS: denial-of-service, e.g. syn flood.
# * R2L: unauthorized access from a remote machine, e.g. guessing password.
# * U2R:  unauthorized access to local superuser (root) privileges, e.g., various buffer overflow attacks.
# * probing: surveillance and other probing, e.g., port scanning.
#
#
# Source of the dataset : https://www.unb.ca/cic/datasets/nsl.html
#
# usefull links:
# - https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
# - https://github.com/dimtics/Network-Intrusion-Detection-Using-Machine-Learning-Techniqu
# - https://github.com/imRP26/Network-based-Intrusion-Detection-Systems
# - https://nycdatascience.com/blog/student-works/network-intrusion-detection/
# - https://www.youtube.com/watch?v=M50pQfj9ZOI&feature=youtu.be

# <h2>1.4. Real-world/Business objectives and constraints.</h2>

# * No low-latency requirement.
# * Interpretability partially is important.
# * Intrusion Detection should not take hours.It should fininsh in a few seconds or a minute.
# * It should detect the network intrusion as well as possible.

# # 2 : Machine learning problem formulation

# ## 2.1 : Data

# ### 2.1.1 : Data Overview

# Source of the data : https://www.unb.ca/cic/datasets/nsl.html
#
# we have 2 dataset
#  - Train data : It has 125973 datapoints with 42 features
#  - Test data  : it has 22544 datapoints with 42 features
#
# here is a detailed description about the dataset
# http://kdd.ics.uci.edu/databases/kddcup99/task.html

# ## example data point

# <pre>
# duration                        -      0
# protocol_type                   -      tcp
# service	                     -      ftp_data
# flag                            -      SF
# src_bytes                       -      491
# dst_bytes                       -      0
# land                            -      0
# wrong_fragment                  -      0
# urgent                          -      0
# hot	                         -      0
# num_failed_logins               -      0
# logged_in                       -      0
# num_compromised                 -      0
# root_shell                      -      0
# su_attempted                    -      0
# num_root                        -      0
# num_file_creations              -      0
# num_shells                      -      0
# num_access_files                -      0
# num_outbound_cmds               -      0
# is_host_login                   -      0
# is_guest_login                  -      0
# count                           -      2
# srv_count	                   -      0.0
# serror_rate	                 -      0.0
# srv_serror_rate	             -      0.0
# rerror_rate                     -      0.0
# srv_rerror_rate                 -      1.0
# same_srv_rate	               -      0.0
# diff_srv_rate                   -      0.0
# srv_diff_host_rate              -      150
# dst_host_count	              -      25
# dst_host_srv_count              -      0.17
# dst_host_same_srv_rate	      -      0.03
# dst_host_diff_srv_rate	      -      0.17
# dst_host_same_src_port_rate	 -      0.0
# dst_host_srv_diff_host_rate	 -      0.0
# dst_host_serror_rate            -      0.0
# dst_host_srv_serror_rate        -      0.0
# dst_host_rerror_rate	        -      0.05
# dst_host_srv_rerror_rate	    -      0.0
# attack                          -      normal
# </pre>

# <h2>2.2. Mapping the real-world problem to an ML problem</h2>

# <h3>2.2.1. Type of Machine Learning Problem</h3>

# <pre>
# There are 2 type of class we need to classify attack or normal -> This is a binary classification task
# </pre>

# <h3>2.2.2. Performance Metric</h3>

# <pre>
# some of the research papers and solution have been used this metric
# * AUC
# * f1 score
# lets use this metric also to get some interpretability
# * Binary Confusion matrix
# * Detection rate - It it nothing but the recall
# </pre>

# <h3>2.2.3. Machine Learing Objectives and Constraints</h3>

# <pre>
# Objective : Given a datapoint classfy is it an attack or not -> Binary Classification
#
# Constraints:
#     1. reasonable latency
#     2. Interpritability
# </pre>

# ### 2.2.4 Train Test Datasets
# we already have train and test data set from the source.

# <h1>3. Exploratory Data Analysis</h1>

# <h2>3.1 Reading the data</h2>

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.utils import resample
from mlxtend.classifier import StackingClassifier

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# to display all column of datapoints
pd.set_option('display.max_columns', None)


# ### 3.1.1  Reading train data

# In[3]:


# reading the train data
# giving feature name expliticly as in the train data these are missing
fetaures_name =  ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]


# please specify the sep = ',' parameter ,else all the datapoints will placed in the first column itself
train_data =  pd.read_table("KDDTrain+.txt",sep = ',', names=fetaures_name)
train_data.head()


# In[4]:


# there is an extra feature present at 43 number column which is not useful remove it.
# for this lets use iloc : integer location , where we will do indexing for selection by position
train_data = train_data.iloc[:,:-1]
train_data.head()


# In[5]:


print("Shape of the training data",train_data.shape)
print("number of data points ",train_data.shape[0])
print("Number of feature ",train_data.shape[1])


# In[6]:


train_data.info()


# ### 3.1.2 : gving class label to the attacks
# ##### Normal   : 0
# ##### all other attack : 1

# In[7]:


# creating a function to give label
def labeling(x):
    if x == 'normal':
        return 0
    else:
        return 1

#stroing all the attack in the variable label
label = train_data['attack']

# mapping all the attack to the desired output which is 0 and 1
class_label = label.map(labeling)

#creating a new column called label in the training data
train_data['label'] = class_label


# In[8]:


print("shape of the train data",train_data.shape)
train_data.head(3)


# In[9]:


# distribution plot of class 1 and class 0
f, ax = plt.subplots(figsize=(12,6))
ax = sns.countplot(x = 'label' , data = train_data , hue = 'label')
plt.title("Normal and attack distribution")
plt.grid()
plt.show()


# ### 3.1.3 Reading Test data

# In[10]:


# reading test data
test_data =  pd.read_table("KDDTest+.txt",sep = ',', names=fetaures_name)
test_data.head()


# In[11]:


#removing extra useless feature
test_data = test_data.iloc[:,:-1]
test_data.head()


# In[12]:


print("Shape of the test data",test_data.shape)
print("number of data points ",test_data.shape[0])
print("Number of feature ",train_data.shape[1])


# In[13]:


test_data.info()


# ### 3.1.4 : gving class label to the attacks
# ##### Normal   : 0
# ##### all other attack : 1

# In[14]:


#stroing all the attack in the variable label
label = test_data['attack']

# mapping all the attack to the desired output which is 0 and 1
class_label = label.map(labeling)

#creating a new column called label in the training data
test_data['label'] = class_label


# In[15]:


print("shape of the test data",test_data.shape)
test_data.head(3)


# In[16]:


#distribution plot of class 0 and class 1
f, ax = plt.subplots(figsize=(12,6))
ax = sns.countplot(x = 'label' , data = test_data , hue = 'label')
plt.title("Normal and attack distribution")
plt.grid()
plt.show()


# ### Observation

# - There are 42 features both in train and test dataset
#
#
# - 15 float value , 23 integer value and 4 object value
#
#
# - its look like we dont have null value , however we will recheck again. $
#
#
# - In the distribution plot of class 0 and 1 : In train dataset class 0 has more datapoints than class 1 and in test dataset class1 has more datapoints than class 0

# ## 3.2 Data Cleaning

# ##### Checking for duplicates values

# In[17]:


# drop_duplicates () : this function return DataFrame with duplicate rows removed.
train_data = train_data.drop_duplicates(subset = fetaures_name[:-1] , keep ='first' , inplace = False)
train_data.shape


# In[38]:


##### Checking for Null values


# In[19]:


null_rows = train_data[train_data.isnull().any(1)]
print(null_rows)


# #### Observation
# - no duplicate values present
# - we dont have null values

# ## 3.3 Distribution of attacks in the dataset

# ### 3.3.1 : Train data

# In[20]:


# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(25,15))

# we need the total number of data to find the percentage later.
total = len(train_data) * 1

# below code will simply plot bar plot where X axis is attack(23 classes) and y will simply count
ax = sns.countplot(x="attack", data=train_data)


# each p of patches(which is from the countplot) has height(number of data point for a given class ),width.
# then pass p to annotate(it is used to show text) and computing % of data in each class , give x coord and y coord of rectangle
for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x(), p.get_height()))


# In th yaxis we are giving interval(11 interval) of datapoints
ax.yaxis.set_ticks(np.linspace(0, total, 11))


# adjust the ticklabel to the desired format, without changing the position of the ticks.
# map() need the function(what to do) and iterative
# below code : ax.yaxis.get_majorticklocs() - it will give 11 value from 0 to 125973 and we are dividing it with the total value which is 125973 after that we are getting 11 intervals with the percentile 
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))

plt.title("Distribution of y_i in training data")
plt.grid()
plt.show()


#
# ##### datapoint belonging to each class

# In[21]:


# take the all the class with datapoints belonging to each classes and sort them by label
train_class_distribution = train_data['attack'].value_counts()

# it is sorting them in decreasing order (by number of datapoints)
sorted_yi = np.argsort(-train_class_distribution.values)
# now for each i of the sorted datapoints we are printing the number of datapoints and the percetange
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/len(train_data)*100), 3), '%)')


# ### Observation :
# In the above plot we have 23 different kind of attacks and their distributions :
#
# - data set is not uniform distributed as we can see in the above
#
#
# - there are lots of attacks where data points are very few and some of the attacks like normal and neptune these both have 85% datapoints out of 100% datapoints    
#
#
# - There are 16 attacks out of 23 attacks where the data points are less then 1%
#

# ##### From this we see that : Normal has 53.5% datapoints and all other 22 class has 46.5% datapoints
# we got an imbalanced dataset

# ### 3.3.2 Test data

# In[22]:


#Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(40,20))

# we need the total number of data to find the percentage later.
total = len(test_data) * 1

# below code will simply plot bar plot where X axis is attack(23 classes) and y will simply count
ax = sns.countplot(x="attack", data= test_data )

# each p has height(number of data point for a given class ),width.
# then pass p to annotate(it is used to show text) and computing % of data in each class , give x coord and y coord of rectangle
for p in ax.patches:
       ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x(), p.get_height()))

# put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
ax.yaxis.set_ticks(np.linspace(0, total, 11))

# adjust the ticklabel to the desired format, without changing the position of the ticks.
# map() need the function(what to do) and iterative
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
plt.grid()
plt.title("Distrubtion of y_i in test data")
plt.show()


# In[23]:


# take the all the class with datapoints belonging to each classes and sort them by label
test_class_distribution = test_data['attack'].value_counts()

# it is sorting them in decreasing order (by number of datapoints)
sorted_yi = np.argsort(-test_class_distribution.values)
# now for each i of the sorted datapoints we are printing the number of datapoints and the percetange
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',test_class_distribution[i], '(', np.round((test_class_distribution.values[i]/len(test_data)*100), 3), '%)')


# - there is an intresting thing that is in the test data we have 38 classes.Which did not come to notice earlier.
#
# - here also Normal and naptune classes has larger number of datapoints
# -  the same story data is not uniform
# - dataset is imbalanced

# ### 3.2.3  attacks which are not in train data

# In[24]:


# put train and test attack ina set and just find the difference we will get the classes.
trn = set(train_data['attack'].unique())
tst = set(test_data['attack'].unique())

extra = tst - trn

print(extra)
print("*"*100)
print("number of extra Attacks : ",len(extra))


# In[25]:


e_extra = tst - extra
print(e_extra)
print("*"*100)
print("Attacks which are present in both train and test data ", len(e_extra))


# In[26]:


ee_extra = trn - e_extra
print("Attacks which are present in train and not in  test data",ee_extra)


# ### Observation
# - we have 17 extra classes in the test data
# - 21 attacks present both in testa nd train dataset
# - there are 2 classes which are not present in the test data but present in the train data , namely : 'spy', 'warezclient  

# ## 3.4 Univariate analysis on catagorical features

# ### 3.4.1 Univariate analysis on protocol_type

# #### [i] How many category present in this feature

# In[27]:


unique_proto = train_data['protocol_type'].value_counts()
print("Number of unique proto type : ", unique_proto.shape[0])
print(unique_proto)


# #### Observation
# - we have 3 different category of proto type in the training data namely : TCP , UDP and ICMP
# - lots of points belongs to tcp where as udp and icmp has fewer points

# #### [ii] Distribution of the this feature

# In[28]:


f, ax = plt.subplots(figsize=(12,6))
ax = sns.countplot(x = 'protocol_type' , data = train_data , hue = 'label')
plt.title("Normal and attack distribution in protocolo_type feature")
plt.grid()
plt.show()


# #### Observation
# - There are lots of point from the training data belongs to tcp prtocol_type (102689) . Normal and attack classes both are uniform only in term of tcp. 
# - majority of the udp protoype belongs to normal class while there are few points belongs to attack class also.
# - In icmp protocol_type majority of points belongs to attack class .

# #### [iii] Featurizing using one hot encoding

# In[20]:


prototype_vectorizer = CountVectorizer()
train_protocol_type_encoding = prototype_vectorizer.fit_transform(train_data['protocol_type'])
test_protocol_type_encoding = prototype_vectorizer.transform(test_data['protocol_type'])


# In[25]:


print("train_protocol_type_encoding is converted feature using one-hot encoding method. The shape of gene feature:",train_protocol_type_encoding.shape)


# #### [iv] How good is this protocol_type feature in predicting y_i ?
# To answer this question will build a Decision tree model model using only protocol_type feature (one hot encoded) to predict y_i.

# In[18]:


# defining y_ture , y_test
y_true = train_data['label']
y_test = test_data['label']


# In[19]:


# Initializatioin of hyperparam and lets take only two hyperparam to tune
parameters = {'max_depth':[1, 5, 10, 50, 100, 500, 1000],
              'min_samples_split':[5, 10, 100, 500]}

# using grid search lets find out the best hyperparam value
# Decision tree using gini impurity
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
DT_bow = GridSearchCV(DTC(criterion= 'gini'), parameters, cv=3 ,scoring='roc_auc')
DT_bow.fit(train_protocol_type_encoding,y_true)

# cv_results_dict of numpy (masked) ndarrays
# it will give mean train score as an array
cv_auc = DT_bow.cv_results_['mean_train_score']

max_depth = [1,5,10,50,100,500,1000]
min_samples_split = [5,10,100,500]


# reshaping the array (cv_auc) into a shape of (7,4)
# reference:https://qiita.com/bmj0114/items/8009f282c99b77780563
scores = cv_auc.reshape(len(max_depth),len(min_samples_split))
plt.figure(figsize = (12,6))
df = pd.DataFrame(scores, index=max_depth, columns=  min_samples_split)
sns.heatmap(df, annot=True)


# In[20]:


DT_bow = DTC(criterion= 'gini', max_depth = 10 , min_samples_split =10 )
DT_bow.fit(train_protocol_type_encoding , y_true)

# roc_curve function will return 3 thing fpr,trp, threshold
# calling predict_proba with the best estimater that we have
# train fpr and tpr give the an arry with flauctuate value
train_fpr, train_tpr, thresholds = roc_curve(y_true, DT_bow.predict_proba(train_protocol_type_encoding)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, DT_bow.predict_proba(test_protocol_type_encoding)[:,1])

# auc() : this function will give area under the curve value : using somtehing called Trapezoidal_rule
# to know more about this link :https://en.wikipedia.org/wiki/Trapezoidal_rule
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[21]:


# to find important feature create a dataframe
# pass data where we have the DTC attribute feature importance will give all the feature with important feature
# next pass the index where it will have the feature name coresponding to the feature importance value
# sort all the value desceding order

importance_feature = pd.DataFrame(data = DT_bow.feature_importances_.T, index = prototype_vectorizer.get_feature_names()).sort_values(by = 0, ascending=False)[:20]
print("Top 20 important features :",importance_feature)


# ### Observation
# - by using only this feature i am getting 58 auc score from this we get to know that this feature may helpful in predicting the yi when we build the actual model using all features.
# - the modeling thinking that most important category is udp where as icmp is not at all important

# ### 3.4.2 Univariate analysis on service

# #### [i] How many category present in this feature

# In[31]:


unique_service = train_data['service'].value_counts()
print("Number of unique service : ",unique_service.shape[0])
print(unique_service.head())


# #### Observation
# - there are two services http and private has quite more datapoints than other.

# #### [ii] Distribution of the this feature

# In[33]:


s = sum(unique_service.values)
h = unique_service.values/s
plt.plot( h , label = 'Histogram of service')
plt.xlabel('index of a service')
plt.ylabel('Number of occurance')
plt.legend()
plt.grid()
plt.show()


# #### Observation
# - this is a skewed distribution
# - there are few services occur more and major of service occur less time .
# - In this distribution from left to right services in a decreasing order (frequency).
# - 0th index contain the http, 1st index contain private etc.

# In[34]:


c = np.cumsum(h)
plt.plot(c , label = "Cumulative Distribution of service")
plt.grid()
plt.legend()
plt.show()


# #### Observation
# - Top 20 to 25 services contributed to 90 percent of data.that means these services occur very frequently than other services.
#

# #### [iii] Featurizing using one hot encoding

# In[21]:


service_encode = CountVectorizer()
train_service_encoding = service_encode.fit_transform(train_data['service'])
test_service_encoding = service_encode.transform(test_data['service'])


# In[29]:


print("train_service_encoding is converted feature using one-hot encoding method. The shape of gene feature:",train_service_encoding.shape)


# #### [iv] How good is this protocol_type feature in predicting y_i ?
# To answer this question will build a Decision tree model model using only protocol_type feature (one hot encoded) to predict y_i.

# In[31]:


# Initializatioin of hyperparam and lets take only two hyperparam to tune
parameters = {'max_depth':[1, 5, 10, 50, 100, 500, 1000],
              'min_samples_split':[5, 10, 100, 500]}

# using grid search lets find out the best hyperparam value
# Decision tree using gini impurity
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
DT_bow = GridSearchCV(DTC(criterion= 'gini'), parameters, cv=3 ,scoring='roc_auc')
DT_bow.fit(train_service_encoding,y_true)

# cv_results_dict of numpy (masked) ndarrays
# it will give mean train score as an array
cv_auc = DT_bow.cv_results_['mean_train_score']
max_depth = [1,5,10,50,100,500,1000]
min_samples_split = [5,10,100,500]


# reshaping the array (cv_auc) into a shape of (7,4)
# reference:https://qiita.com/bmj0114/items/8009f282c99b77780563
scores = cv_auc.reshape(len(max_depth),len(min_samples_split))

plt.figure(figsize = (12,6))
df = pd.DataFrame(scores, index=max_depth, columns=  min_samples_split)
sns.heatmap(df, annot=True)


# In[35]:


from sklearn.metrics import roc_curve, auc

DT_bow = DTC(criterion= 'gini', max_depth = 5 , min_samples_split = 5 )
DT_bow.fit(train_service_encoding , y_true)

# roc_curve function will return 3 thing fpr,trp, threshold
# calling predict_proba with the best estimater that we have
# train fpr and tpr give the an arry with flauctuate value
train_fpr, train_tpr, thresholds = roc_curve(y_true, DT_bow.predict_proba(train_service_encoding)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, DT_bow.predict_proba(test_service_encoding)[:,1])


# auc() : this function will give area under the curve value : using somtehing called Trapezoidal_rule
# to know more about this link :https://en.wikipedia.org/wiki/Trapezoidal_rule
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[37]:


# to find important feature create a dataframe
# pass data where we have the DTC attribute feature importance will give all the feature with important feature
# next pass the index where it will have the feature name coresponding to the feature importance value
# sort all the value desceding order

importance_feature = pd.DataFrame(data = DT_bow.feature_importances_.T, index=service_encode.get_feature_names()).sort_values(by = 0, ascending=False)[:10]
print("Top 20 important features :",importance_feature)


# #### Observation
# - by using only this single feature my model giving 87 test auc value which is quite intersting
# - Out of 70 feature only 5 of them is important .
# - This information might be useful in feature engineering (we can remove feature with 0.0 values)

# ### 3.4.3 Univariate analysis on Flag

# #### [i] How many category present in this feature

# In[34]:


flag_unique = train_data['flag'].value_counts()
print("Number of unique flag : ", flag_unique.shape[0])
print(flag_unique.head())


# #### [ii] Distribution of the this feature

# In[38]:


# taking sum
s = sum(flag_unique.values)
# diving each falg vaue to sum
h = flag_unique.values/s
plt.plot(h , label = 'Histogram of flag')
plt.xlabel("index of flag")
plt.ylabel('Number of occurance')
plt.grid()
plt.legend()
plt.show()


# - there are 3 - 4 falg has mpre number of occurance
# - skewed distribution

# In[39]:


c = np.cumsum(h)
plt.plot(c , label = 'Cumulative Distribution of flag')
plt.legend()
plt.grid()
plt.show()


# - out of 10 falg 4 flag contibuted 98 -99% of data , these 4 flags are occuring more freuntly.

# #### [iii] Featurizing using one hot encoding

# In[22]:


flag_encoding = CountVectorizer()
train_flag_encoding = flag_encoding.fit_transform(train_data['flag'])
test_flag_encoding = flag_encoding.transform(test_data['flag'])


# In[27]:


print("train_flag_encoding is converted feature using one-hot encoding method. The shape of gene feature:",train_flag_encoding.shape)


# #### [iv] How good is this protocol_type feature in predicting y_i ?
# To answer this question will build a Decision tree model model using only protocol_type feature (one hot encoded) to predict y_i.

# In[40]:


# Initializatioin of hyperparam and lets take only two hyperparam to tune

parameters = {'max_depth':[1, 5, 10, 50, 100, 500, 1000],
              'min_samples_split':[5, 10, 100, 500]}

# using grid search lets find out the best hyperparam value
# Decision tree using gini impurity
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
DT_bow = GridSearchCV(DTC(criterion= 'gini'), parameters, cv=3 ,scoring='roc_auc')
DT_bow.fit(train_flag_encoding,y_true)

# cv_results_dict of numpy (masked) ndarrays
# it will give mean train score as an array
cv_auc = DT_bow.cv_results_['mean_train_score']
max_depth = [1,5,10,50,100,500,1000]
min_samples_split = [5,10,100,500]


# reshaping the array (cv_auc) into a shape of (7,4)
# reference:https://qiita.com/bmj0114/items/8009f282c99b77780563
scores = cv_auc.reshape(len(max_depth),len(min_samples_split))

plt.figure(figsize = (12,6))
df = pd.DataFrame(scores, index=max_depth, columns=  min_samples_split)
sns.heatmap(df, annot=True)


# In[44]:


from sklearn.metrics import roc_curve, auc

DT_bow = DTC(criterion= 'gini', max_depth = 5 , min_samples_split = 10 )
DT_bow.fit(train_flag_encoding , y_true)

# roc_curve function will return 3 thing fpr,trp, threshold
# calling predict_proba with the best estimater that we have
# train fpr and tpr give the an arry with flauctuate value
train_fpr, train_tpr, thresholds = roc_curve(y_true, DT_bow.predict_proba(train_flag_encoding)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, DT_bow.predict_proba(test_flag_encoding)[:,1])


# auc() : this function will give area under the curve value : using somtehing called Trapezoidal_rule
# to know more about this link :https://en.wikipedia.org/wiki/Trapezoidal_rule
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[45]:


# to find important feature create a dataframe
# pass data where we have the DTC attribute feature importance will give all the feature with important feature
# next pass the index where it will have the feature name coresponding to the feature importance value
# sort all the value desceding order

importance_feature = pd.DataFrame(data = DT_bow.feature_importances_.T, index=flag_encoding.get_feature_names()).sort_values(by = 0, ascending=False)[:11]
print("Top 20 important features :",importance_feature)


# #### Observation
# - By look at the train and  test auc value the model might be overffiting , but we have 78 test auc value, which showing us that this model will helful
# - there is one category which is most important : 'sf' this category itself has value of 95 .

# ## 3.5 Univariate analysis on some continuous features

# ### 1.Duration
# length (number of seconds) of the connection

# In[42]:


plt.figure(figsize =(12,5))
# violin plot
sns.violinplot(x ='label' , y = 'duration' , data = train_data )
plt.show()


# In[43]:


plt.figure(figsize =(12,5))
# violin plot
sns.boxplot(y ='duration' ,  data = train_data )
plt.show()


# #### Observation
# - mean, median, 25th ,50th,75th percentile is so small to analyse beacuse most of the duration is 0
# - lets look into 0 to 100% percentile value

# In[44]:


for i in range(0,100,10):
    # take all the value of duration column
    var = train_data['duration'].values
    # falttend them and sort in ascending order
    var = np.sort(var , axis= None)
    # formula to calculate percentile "int(len(var)*float(i)/100"
    print("{} percentile value {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile value is ",var[-1])


# #### Observation
# - about 90 percentile value of duration is 0
# - there are lots of value we can see in 100 percentile
# - so lets look at from 90 to 100 percentile

# In[45]:


#calculating speed values at each percntile 90,91,92,93,94,95,96,97,98,99,100
for i in range(90,100):
    var = train_data['duration'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print("100 percentile value is ",var[-1])


# In[101]:


#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var = train_data['duration'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# #### Observation
# - value of duration has increased from 90 percentile onward.
# - if we look at the violine plot of class label 1 there is some value which might be upto 2k to 3k
# - from 99.0 to 100 percentile the value is increased dramatically..these value might be outlier.

# ### 2 . src_bytes
# number of data bytes from source to destination

# In[71]:


plt.figure(figsize=(15,8))
sns.violinplot(x = 'label' , y = 'src_bytes' , data = train_data)
plt.show()


# #### Observation
# - for both label 0 and 1 it is hard to analys .but one thing to notice is class 1 which attack has quite larger value than class 0 which is normal 

# In[75]:


plt.figure(figsize = (15,6))
sns.boxplot(y = 'src_bytes' ,data = train_data)
plt.show()


# #### Observation
# - from this box plot we can see all the value from 25th to 75th percentile has zero. it is hard to interprit.
# - lets agaim zoom into the percentile value of src_bytes.

# In[76]:


# CALCULATING PERCENTILE FROM 0,10,20,30,...,100
for i in range(0,100,10):
    var = train_data['src_bytes'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# #### Observation
# - As we can see there is a big jump from 90% to 100 %

# In[77]:


# calculating percentile from 90,91,92,...,100
for i in range(90,100):
    var = train_data['src_bytes'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# In[79]:


#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var = train_data['src_bytes'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# #### Observation
# - there is a huge value in the 100 % which 1379963888byte equivalent to 1.28GB  goes from source to destination.

# ### 3.dst_bytes
# number of data bytes from destination to source

# In[80]:


plt.figure(figsize=(15,8))
sns.violinplot(x = 'label' , y = 'dst_bytes' , data = train_data)
plt.show()


# In[81]:


plt.figure(figsize = (15,6))
sns.boxplot(y = 'dst_bytes' ,data = train_data)
plt.show()


# In[82]:


# CALCULATING PERCENTILE FROM 0,10,20,30,...,100
for i in range(0,100,10):
    var = train_data['dst_bytes'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# In[84]:


# calculating percentile from 90,91,92,...,100
for i in range(90,100):
    var = train_data['dst_bytes'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# In[90]:


#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var = train_data['dst_bytes'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# #### Observation
# - here is also a huge value in the 100 % which 1309937401byte equivalent to 1.21GB  goes from  destination to source.

# ### 4. wrong_fragment
# number of wrong fragments
# IP fragmentation is an Internet Protocol (IP) process that breaks packets into smaller pieces (fragments)

# In[85]:


plt.figure(figsize=(15,8))
sns.violinplot(x = 'label' , y = 'wrong_fragment' , data = train_data)
plt.show()


# #### Observation
# - there is so much variance in the class 1 while there no variance in class 0
# - there is some diffrence in class 0 and 1 .this might help to distinguish from class 1 to 0

# In[86]:


plt.figure(figsize = (15,6))
sns.boxplot(y = 'wrong_fragment' ,data = train_data)
plt.show()


# In[87]:


# CALCULATING PERCENTILE FROM 0,10,20,30,...,100
for i in range(0,100,10):
    var = train_data['wrong_fragment'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# In[88]:


# calculating percentile from 90,91,92,...,100
for i in range(90,100):
    var = train_data['wrong_fragment'].values
    var = np.sort(var , axis = None)
    print("{} percentile value is {}".format(i , var[int(len(var)*float(i)/100)]))
print("100 percentile is ",var[-1])


# In[91]:


#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var = train_data['wrong_fragment'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])


# #### Observation
# - this feature seems to be ok as there is not so much inflection

# ### 3.6 Bivariate Analysis (pair plots)

# In[94]:


n = train_data.shape[0]
sns.pairplot(train_data[['srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate', 'label']][0:n], hue='label', vars=['srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate'])
plt.show()


# #### Observation
# - if we look the dst_host_count and dst_host_same_srv_rate feature there are some point (not fully) but partially separeble ,but there are some overlap point also.
# - dst_hst_srv_count and dst_host_count here also in the middle there are some overlap and some of the datapoints are partially separable 
# - If we look at the pdf of these 4 feature
#         - srv_diff_host_rate : the class 0 have higer value than class 1
#         - dst_host_count : all over the class 1 is placed and have much higher value than class 0
#         - dst_host_srv_count : there is some over lap region between class 1 and 0, class 1 have higher value than class 0.
#         - dst_host_same_srv_rate : it look both class 1 and class 0 separated ,but there are quite overlap datapoints.
# lets try to analyse dst_host_srv_count and dst_host_same_srv_rate

# In[45]:


plt.subplots(figsize = (20,12))
plt.subplot(1,2,1)
sns.violinplot(x = 'label',y = 'dst_host_srv_count',data = train_data )

plt.subplot(1,2,2)
sns.distplot(train_data[train_data['label'] == 1]['dst_host_srv_count'][0:] , label = '1' ,color = 'red' )
sns.distplot(train_data[train_data['label'] == 0]['dst_host_srv_count'][0:] , label = '0' ,color = 'blue' )
plt.show()


# #### Observation
# - These 2 violins are not fully overlap , this "dst_host_srv_count" feature may be usful in classification

# In[47]:


plt.subplots(figsize = (20,12))
plt.subplot(1,2,1)
sns.violinplot(x = 'label',y = 'dst_host_same_srv_rate',data = train_data )

plt.subplot(1,2,2)
sns.distplot(train_data[train_data['label'] == 1]['dst_host_same_srv_rate'][0:] , label = '1' ,color = 'red' )
sns.distplot(train_data[train_data['label'] == 0]['dst_host_same_srv_rate'][0:] , label = '0' ,color = 'blue' )
plt.show()


# #### Observation
# - here also both class is not overlapping fully(in term of 25th 50th and 75th percentile) so there is some separebility in this dst_host_same_srv_rate.
#

# ### 3.7 Multivariate analysis using TSNE

# In[46]:


# Using TSNE lets visualize the data from 32dim(continuous variable) to 2 dim
train_data_sample = train_data[0:7000]
# why minmax ? : actually there is no specific reason beacuse i have tried both and both of them working similarly for this case.
X = MinMaxScaler()
# we have 32 continuous feature.
X  = X.fit_transform(train_data_sample[["duration","src_bytes",
    "dst_bytes","wrong_fragment","urgent","hot","num_failed_logins","num_compromised","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]])
y = train_data_sample['label'].values


# In[47]:


tsne1 = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=2, random_state=None, method='barnes_hut', angle=0.5).fit_transform(X)


# In[51]:


# creating a dataframe by puting a dict : where x will have all the value from 1st col and y will have value of 2nd col
# tsne have emmbeding vector of size (7000,2)
df = pd.DataFrame({'x':tsne1[:,0], 'y':tsne1[:,1] ,'label':y})

# drawing the plot in appropriate place in the grid
# implot is basically a combination of facetgrid and regplot.
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False,palette="Set1",size=8,markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(30, 1000))
plt.show()


# In[62]:


tsne1 = TSNE(n_components=2, perplexity=50.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=2, random_state=None, method='barnes_hut', angle=0.5).fit_transform(X)


# In[63]:


df = pd.DataFrame({'x':tsne1[:,0], 'y':tsne1[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(50, 1000))
plt.show()


# In[53]:


tsne1 = TSNE(n_components=2, perplexity=15.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=2, random_state=None, method='barnes_hut', angle=0.5).fit_transform(X)


# In[54]:


df = pd.DataFrame({'x':tsne1[:,0], 'y':tsne1[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(15, 1000))
plt.show()


# #### Observation
# - Here 32dim continious variable is taken and reducing them to 2 dim
# - from the 3 plot we can observe that thsese 32 continuous variable will gonna be helpful in determining the class label.
# - in the First plot where perplexity is 30 : the class 1 and class 0 are separated , yes there are some region of overlapping but most of them are separeted.The same story is we can see on plot number 2 and in the 3.

# # Summary of EDA

# - After reading the dataset got to know that these dataset have not the feature name in their appropriate column so columns name were given 
#      - Shape of the training data (datapoints : 125973, features : 42)
#      - Shape of the test data (datapoints : 22544, features : 42)
#
#
# - The task is to identify wether a given connection is normal or attack , for that created a column "label" and gave all the attacks which is name normal as class 0 and all other attack as class 1.   
#
#
# - By checking the distribution of the dataset with respect to the class label ,found that dataset is little bit imbalanced(53.5% normal and 46.5% attack).   
#
#
# - Checked for duplicate and null value : there were not null and duplicate value.
#
#
# - Checked for distribution with respect to different attacks in train and test dataset :
#        Train data
#        * data set is not uniform distributed (by looking at different attacks)
#        * there are lots of attacks where data points are very few and some of the attacks like normal and neptune these
#          both have 85% datapoints out of 100% datapoints
#        * There are 16 attacks out of 23 attacks where the data points are less then 1%
#        Test data
#        * look at the test dataset we got bunch of new attacks which are not in the test data
#        * here also the normal and neptune attacks has more datapoints than other
# - To analysis the feature i thought to analyse the categorical feature and numerical feature separetly.
#
#
# - <b>Univariate analysis on categorical feature</b>
#
#      - we have 3 categorical feature : protocol_type , service and flag
#      - to analysis these categorical feature i gone have through 4 things
#
#          1. <b>Number of category present in the dataset </b>:
#          ans :
#             - In protocol_feature : 3 category present tcp,upd and icmp where  majority of the points are from tcp and udp . 
#             - In service feature : 70 unique category present
#             - In flag feature :    11 unique category present
#
#          2. <b>Distribution of the categorical feature</b> :
#         ans :
#             - In protocol_feature : tcp has both normal and attack class datapoints reasonable, where udp has more class0 (normal) points than class1(attack) and icmp has more class 1
#             - In service feature : The distribution is skewed where few services occur more and major of service occur less time.
#             - In flag feature : It is also a skewed distribution
#
#          3. <b>featurization of the categorical feature</b> :
#          ans: All of them have been featurize using one hot encoding
#
#          4. <b>How good is this protocol_type feature in predicting y_i </b>?
#          ans :  building a simple model (Decison tree classifer) for each categorical feature and know the important feature.
#             - In protocol_feature : we get some feature impotance where udp is the most important feature in predicting yi where icmp is less  
#
#             - In service feature : model got test auc value of 87 by only using this feature this means this categorical feature will be useful when we build actual model.There are few feature which is important not all.            
#
#             - In flag feature : this fearure also useful as it has 78 test auc score, the model might overfit a little bit as train and test auc has gap.
#
#
#      - There are some category where the model thought that those are not at all important , we can do some feature engineering by removing those unimportant features.   
#
# - <b> Univariate analysis on continuous feature</b>
#
#      - Duration : majority of points have value of zero of this feature , from 98 percentile onward values are changing.
#
#      - src_bytes : values of this feature are increasing in a slow rate upto 98 - 99.9 , but there is sudden change in the 100 percintile with a quite large value(1.25 gb) which mean 1.25 gb of data goes from source to destination, this might be an outlier.   
#
#      - dst_bytes : this is same as the src_byte , in the 100 percentile there is quite big number(1.21 gb of data from destination to source) this could be an outlier or may be these values are sign of attack.
#      - wrong_fragment : in the the violin plot class 1 has more variance than class 1 while class 0 has value around 0
# - <b> Bivariate Analysis using pair plot </b>
#      - By looking at the pair plot we can say that there are some overlap between class1 and class0, but not fully.
#      - the PDF's of each feature has some information like one pdf(class 0) has more value than another(class1) and vice versa. With this information i looked at the the violin plot and the pdf separetly  of some feature , where some of them are not fully overlap so we can say these fearure may helpful in distinguish
# class0 and class 1
# - <b> Multivariate analysis using tSNE </b>
#      - Here i have taken 32 feature and ploted using tsne with 7k datapoints , there result i get is quite brilliant
#      - classes are separable
#      - less overlapping points.

# <h1>4. Machine Learning Models </h1>

# ###### Machine Learning model as follows :
# - we have fewer features so lets build model which are tend to work well on fewer features.
#
# #### 1. Naive Bayes (Base line model)
#  - Base line model should be simple so that we can compare it with other models .
#
# #### 2. KNN
# #### 3. Logistic regression
#  - Logistic regression beacause , its an experiment may be the line separate well both classes , let see.
#
# #### 4. Decision Tree
# #### 5. Random Forest
# #### 6. Xgboost

# ### Merging all numerical and categorical feature
# In[22]:


# read about hstack : https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html
"""
A = coo_matrix([[1, 2], [3, 4]])
B = coo_matrix([[5], [6]])
hstack([A,B]).toarray()
array([[1, 2, 5],
       [3, 4, 6]])
"""
# take protocol_type one_hot_encoding vector and service one_hot_encoding vector and merge them using hstack
train_protocol_service_encoding = hstack((train_protocol_type_encoding, train_service_encoding))
test_protocol_service_encoding = hstack((test_protocol_type_encoding,  test_service_encoding))

# take train_proto_services_encoding vector and flag one_hot_encoding vector and merge them using hstack
train_protocol_service_flag_encoding = hstack((train_protocol_service_encoding, train_flag_encoding))
test_protocol_service_flag_encoding  = hstack((test_protocol_service_encoding ,test_flag_encoding))

# defining y_train and y_test
y_train = train_data['label']
y_test  = test_data['label']

# removing label, attck,protocol_type,service,flag column from train and test data
train_data.drop(['protocol_type','service','flag','attack','label'], axis=1, inplace=True)
test_data.drop(['protocol_type','service','flag','attack','label'], axis=1, inplace=True)

X_train = hstack((train_protocol_service_flag_encoding , train_data))
X_test = hstack((test_protocol_service_flag_encoding , test_data))


# ### Standardization

# In[23]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean = False)
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# In[24]:


print("Shape of the training data after mergeing - datapoints : ",X_train.shape[0],"features : ",X_train.shape[1],  " and y_train :",y_train.shape[0])
print("Shape of the test data after mergeing - datapoints : ",X_test.shape[0],"features : ",X_test.shape[1]," and y_test : ",y_test.shape[0])


# ### Plot : Confusion matrix ,  Precision , Recall

# In[40]:


# This function plots the confusion matrices given y_i, y_i_hat.'
# refer - AAIC
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column

    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1

    B =(C/C.sum(axis=0))
    #divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]]
    plt.figure(figsize=(20,4))

    labels = [1,2]
    # representing A in heatmap format
    cmap=sns.light_palette("Orange")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")

    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")

    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    plt.show()


# ## 4.1 Base line Model
# ## Naive Bayes
# #### Hyper parameter tuning

# In[48]:


# creating object of multinomial naive bayes
multi_NB = MultinomialNB()

# giving bunch of laplace parameter
parameters = {'alpha': [10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3,10**4]}
alpha_range = [10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3,10**4]

# putting the model in grid search cv to find the best hyper param
clf = GridSearchCV(multi_NB ,parameters,cv = 10 , scoring='roc_auc' , return_train_score = True)

#fitting X_train and y_train with the multinomial naive bayes
clf.fit(X_train, y_train)

train_auc = clf.cv_results_['mean_train_score']
train_auc_std = clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(alpha_range, train_auc, label='Train AUC')
# refer : https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha_range,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha_range, cv_auc, label='CV AUC')
# refer : https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha_range,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.legend()
plt.xlabel("K: hyperparameter")
plt.xscale('log')
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# #### fiiting with best param

# In[55]:


from sklearn.metrics import roc_curve, auc

multi_NB =MultinomialNB(alpha = 0.01)
multi_NB.fit(X_train,y_train)


train_fpr, train_tpr, thresholds = roc_curve(y_train, multi_NB.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, multi_NB.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))


plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[28]:


y_train_pred = multi_NB.predict(X_train)
y_test_pred = multi_NB.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
print("*"*100)
print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[57]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# ###### Before getting to observation lets know how to read these metric
# - what test auc score 64 means :- chance of classfying points correctly is 64%
#
# - F1 score - it is the inverse of avg of precision and recall . it will give high value when both precision and recall is high  
#
# - reading precsion and reacll matrix :-
#
#    - precision (columns sums to 1) : of all the point which are predicted to belong to class0 67% are actually belong to class 0 and 33% are belong to class1 
#
#    - recall(row sums to 1) : of all the point which are actually belong to class1 63% are predicted to class 1 and 36%  class0  
#
# ### Observation :
# - there is a gap in trian and test AUC value which mean the model is overfitting
# - lets do some feature selection to reduce the overfitting
# - feature selection by Recursive feature elemination

# ## 4.2 Feature selection by recursive feature elemination

# Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.  
# refer : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# In[26]:


## for the whole feature selection section please refer these two link which i have mentioned below
### refer : https://github.com/dimtics/Network-Intrusion-Detection-Using-Machine-Learning-Techniqu
### refer : https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15

#Encoding the categorical data using label enocder
encoder = LabelEncoder()

# get the categorical features
cat_train = train_data.select_dtypes(include = 'object').copy()
cat_test = test_data.select_dtypes(include = 'object').copy()

cat_train_encode = cat_train.apply(encoder.fit_transform)
cat_test_encode = cat_test.apply(encoder.fit_transform)

# dropping attack feature
cat_train_encode = cat_train_encode.drop(['attack'],axis = 1)
cat_test_encode = cat_test_encode.drop(['attack'],axis = 1)


# In[27]:


# shape of the encoded feature
print(cat_train_encode.shape)
print(cat_train_encode.shape)


# In[28]:


# get all the numerical feature
num_train = train_data.select_dtypes(include = ['float64','int64'])
num_test = test_data.select_dtypes(include = ['float64','int64'])

# join numerical and categorical feature
features = pd.concat([num_train ,cat_train_encode],axis =1).columns
x_train_encode = np.concatenate((num_train , cat_train_encode),axis=1)
x_test_encode = np.concatenate((num_test , cat_test_encode),axis=1)


# In[29]:


# shape after joining categorical and numerical features
print(x_train_encode.shape)
print(x_train_encode.shape)


# In[30]:


# converting the joined feature into a dataframe
x_train_encoder = pd.DataFrame(x_train_encode ,columns = features)
x_test_encoder = pd.DataFrame(x_test_encode , columns = features)

# dropping the label class
x_train_encoder = x_train_encoder.drop(['label'],axis = 1)
x_test_encoder = x_test_encoder.drop(['label'],axis = 1)


# In[31]:


# shape after removing label
print(x_train_encoder.shape)
print(x_test_encoder.shape)


# In[32]:


# this feature is useless as it has only contain zeros ,remove from dataframe
x_train_encoder = x_train_encoder.drop(['num_outbound_cmds'],axis = 1)
x_test_encoder = x_test_encoder.drop(['num_outbound_cmds'],axis = 1)


# In[33]:


# print shape
print(x_train_encoder.shape)
print(x_test_encoder.shape)


# In[36]:


# put correlated features into set , beacuse it will store only single feature not redundant
correlated_features = set()

# create a correlated feature of the train data
# it Compute pairwise correlation of columns, excluding NA/null values.
# by default "pearson correlation "
# refer : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
correlation = x_train_encoder.corr()

#iterate through each column
for i in range(correlation.shape[0]):
    #iterate through ecah value of the given column
    for j in range(i):
        # give the i : column and j : vlaue in that column
        # if the value is > .8 remove them
        if abs(correlation.iloc[i,j]) > 0.8:
            # take that column which is >.8
            column = correlation.columns[i]
            # add it to the above set
            correlated_features.add(column)



# In[37]:


# print which are irrelavant features and how many them
print(correlated_features)
print(len(correlated_features))


# In[38]:


# droppin the irrelevant feature
x_train_encoder = x_train_encoder.drop(['is_guest_login', 'dst_host_srv_rerror_rate', 'srv_serror_rate', 'num_root', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'srv_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate'],axis=1)


# In[41]:


# shape after dropping the irrelevant fetaures
x_train_encoder.shape


# In[43]:


# create a randomforest classifer (why :  beacuse random forest tend to work well on feature importance)
rfc = RandomForestClassifier(random_state=101)

# put the object of randomforest into the recursive feature elemination using cross validation
# refer : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='roc_auc')
rfecv.fit(x_train_encoder, y_train)


# In[44]:


print('Optimal number of features: {}'.format(rfecv.n_features_))


# In[45]:


plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='orange', linewidth=3)
plt.show()


# In[49]:


dset = pd.DataFrame()
dset['attr'] = x_train_encoder.columns

dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)
print(dset.attr)

plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='orange')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()


# ##### Understanding the above 4.2 section :
# - take the object type feature (means categorical) encode using label encoder
# - merge both encoder categorical features and numerical features
# - compute the correlation matrix using a dataframe method corr() which by default use pearson correlation coefficient
# - iterate thorugh two for loop and take out those feature whose value greater than .8 int he correlation matrix
# - now by using recursive feature elemination (using random forest model , we can use any model of our choice but RF give good feature importance) it gave 29 feature which are useful in predicting the model(called optimal features)
# - Then plot the most important features by looking we can again remove some feature which are vey small value

# #### lets use those selected features

# In[23]:


# pleae re run the first few cell
# removing correlated features/irrelevant features from train data
X_train_after_FS = train_data.drop(['srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'is_guest_login', 'dst_host_srv_serror_rate', 'num_root', 'dst_host_serror_rate', 'srv_rerror_rate'],axis = 1)
X_test_after_FS = test_data.drop(['srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'is_guest_login', 'dst_host_srv_serror_rate', 'num_root', 'dst_host_serror_rate', 'srv_rerror_rate'],axis = 1)


# In[24]:


# shape after removing correlated features/irrelevant features from traing data
print(X_train_after_FS.shape )
print(X_test_after_FS.shape )


# In[25]:


# removing featues which are adding less value , by looking at the RFECV
X_train_after_FS = X_train_after_FS.drop(['srv_diff_host_rate', 'num_file_creations' , 'num_failed_logins','num_access_files','root_shell','num_shells','su_attempted','urgent','land', 'num_outbound_cmds','attack'],axis = 1)
X_test_after_FS = X_test_after_FS.drop(['srv_diff_host_rate', 'num_file_creations' , 'num_failed_logins','num_access_files','root_shell','num_shells','su_attempted','urgent','land', 'num_outbound_cmds','attack'],axis = 1)


# In[26]:


# shape after removing above features
print(X_train_after_FS.shape )
print(X_test_after_FS.shape )


# In[27]:


# defining y_train and y_test
y_train = train_data['label']
y_test  = test_data['label']

# removing label, attck,protocol_type,service,flag column from train and test data
X_train_after_FS.drop(['protocol_type','service','flag','label'], axis=1, inplace=True)
X_test_after_FS.drop(['protocol_type','service','flag','label'], axis=1, inplace=True)

print(X_train_after_FS.shape)
print(X_train_after_FS.shape)


# In[28]:


# take protocol_type one_hot_encoding vector and service one_hot_encoding vector and merge them using hstack
train_protocol_service_encoding = hstack((train_protocol_type_encoding, train_service_encoding))
test_protocol_service_encoding = hstack((test_protocol_type_encoding,  test_service_encoding))

# take train_proto_services_encoding vector and flag one_hot_encoding vector and merge them using hstack
train_protocol_service_flag_encoding = hstack((train_protocol_service_encoding, train_flag_encoding))
test_protocol_service_flag_encoding  = hstack((test_protocol_service_encoding ,test_flag_encoding))


# In[29]:


# merging the categorical onehot encoded feature and numerical feature
X_train = hstack((train_protocol_service_flag_encoding , X_train_after_FS))
X_test = hstack((test_protocol_service_flag_encoding , X_test_after_FS))


# In[30]:


print("Shape of the training data after mergeing - datapoints : ",X_train.shape[0],"features : ",X_train.shape[1],  " and y_train :",y_train.shape[0])
print("Shape of the test data after mergeing - datapoints : ",X_test.shape[0],"features : ",X_test.shape[1]," and y_test : ",y_test.shape[0])


# In[31]:


# standardization
scalar = StandardScaler(with_mean = False)
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# ## Modeling with selected features

# ## 4.3 Naive Bayes with Hyperparameter tuning

# In[66]:


# creating object of multinomial naive bayes
multi_NB = MultinomialNB()

# giving bunch of laplace parameter
parameters = {'alpha': [10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3,10**4]}
alpha_range = [10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3,10**4]

# putting the model in grid search cv to find the best hyper param
clf = GridSearchCV(multi_NB ,parameters,cv = 10 , scoring='roc_auc' , return_train_score = True)

#fitting X_train and y_train with the multinomial naive bayes
clf.fit(X_train, y_train)

train_auc = clf.cv_results_['mean_train_score']
train_auc_std = clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(alpha_range, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha_range,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha_range, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha_range,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.legend()
plt.xlabel("K: hyperparameter")
plt.xscale('log')
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[67]:


multi_NB =MultinomialNB(alpha = 10)
multi_NB.fit(X_train,y_train)


train_fpr, train_tpr, thresholds = roc_curve(y_train, multi_NB.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, multi_NB.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))


plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[67]:


y_train_pred = multi_NB.predict(X_train)
y_test_pred = multi_NB.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
print("*"*100)
print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[71]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - there is a slight improvement on the f1 score and the recall aswell after removing some irrelavent features
# - there seems to be some confusion on recall : of all the actual point 64% are predicted to be class 2 and around 36% predicted to be class 0

# ## 4.4 KNN Hyperparameter tuning

# In[52]:


from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

alpha = [5, 11, 15, 21, 31, 41, 51, 99]

auc = []
for i in alpha:
    clf = KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train, y_train)
    predict_y = clf.predict_proba(X_test)[:,1]
    auc.append(roc_auc_score(y_test, predict_y))
    print('For values of alpha = ', i, "The auc score is:",roc_auc_score(y_test, predict_y))

fig, ax = plt.subplots()
ax.plot(alpha, auc,c='orange')
for i, txt in enumerate(np.round(auc,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],auc[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()


# In[54]:


clf = KNeighborsClassifier(n_neighbors =99)
clf.fit(X_train, y_train)
predict_y_train = clf.predict_proba(X_train)[:,1]
print('For values of best alpha = ', 99, "The train auc score is:",roc_auc_score(y_train, predict_y_train))
predict_y_test = clf.predict_proba(X_test)[:,1]
print('For values of best alpha = ', 99, "The test auc score is:",roc_auc_score(y_test, predict_y_test))


# In[55]:


y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)


# In[56]:


print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
print("*"*100)
print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[59]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - The simple knn model giving us more auc as well as f1 and recall value which is good sign
# - but still there is some confusion on test data between class 2 and class 1 in the recall
# - This may be because of the class imbalance

# ## 4.5 Logistic regression Hyperparameter tuning

# In[161]:


# giving range of hyperparam value which we wanted to try to find the best param
parameters = [{'C': [10**-4 , 10**-3 , 10**-2 , 10**-1 , 10**0, 10**1 , 10**2, 10**3, 10**4]}]

# to find best param use grid search or random search this upto you
clf = GridSearchCV(LogisticRegression(penalty = 'l1'), parameters , cv=3 ,scoring='roc_auc')
# fitting it to the train data
clf.fit(X_train,y_train)

# these below 3 line code will give train,test mean and standard deviation value
train_auc = clf.cv_results_['mean_train_score']
train_auc_std = clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(lambda_range, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(lambda_range,train_auc - train_auc_std,train_auc + train_auc_std , alpha=0.2, color='darkblue')

plt.plot(lambda_range, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(lambda_range,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
plt.legend()
plt.xlabel("lmbda_range: hyperparameter")
plt.xscale("log")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[173]:


best_lambda = 0.01
LG = LogisticRegression(C = best_lambda , penalty = 'l1',class_weight={0:.1,1:.15})
LG.fit(X_train , y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, LG.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, LG.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[174]:


from sklearn.metrics import f1_score

y_train_pred = LG.predict(X_train)
y_test_pred = LG.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))


# In[177]:


print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[175]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - The model has high auc value but the f1 and recall is lower than the base line model
# - here also at recall model has some confusion

# ## 4.6 Decision tree with hyperparameter tuning

# In[78]:


# Initializatioin of hyperparam and lets take only two hyperparam to tune
from scipy.stats import randint as sp_randint
parameters =  parameters = {'max_depth':[1, 5, 10, 50, 100, 500, 1000],
              'min_samples_split':[5, 10,50,100, 500]}
# using grid search lets find out the best hyperparam value
# Decision tree using gini impurity
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
DT = GridSearchCV(DTC(criterion= 'gini'), parameters, cv=3 ,scoring='roc_auc')
DT.fit(X_train,y_train)

print('mean test scores',rf_random.cv_results_['mean_test_score'])
print('mean train scores',rf_random.cv_results_['mean_train_score'])


# In[79]:


print('mean test scores',DT.cv_results_['mean_test_score'])
print('mean train scores',DT.cv_results_['mean_train_score'])


# In[80]:


print(DT.best_estimator_)


# In[89]:


DT =DTC(class_weight={0:.1,1:15}, criterion='gini', max_depth=100,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=500,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')


# In[90]:


from sklearn.metrics import roc_curve, auc

DT.fit(X_train , y_train)

# roc_curve function will return 3 thing fpr,trp, threshold
# calling predict_proba with the best estimater that we have
# train fpr and tpr give the an arry with flauctuate value
train_fpr, train_tpr, thresholds = roc_curve(y_train, DT.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, DT.predict_proba(X_test)[:,1])


# auc() : this function will give area under the curve value : using somtehing called Trapezoidal_rule
# to know more about this link :https://en.wikipedia.org/wiki/Trapezoidal_rule
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[95]:


y_train_pred = DT.predict(X_train)
y_test_pred = DT.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))


# In[97]:


print('Train recall score or detection rate',recall_score(y_train,y_train_pred))
print('Train recall score or detection rate',recall_score(y_test,y_test_pred))


# In[92]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - This is so far the best model we have
# - after doing some class balance we have got pretty good auc , f1 score and recall also
# - now model not confused as previous on class 1 and class 0

# ## 4.7 Random Forest Hyperparameter tuning

# In[71]:


param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(110,190),
              "min_samples_leaf": sp_randint(25,65)}

clf = RandomForestClassifier(random_state=25,n_jobs=-1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=10,scoring='roc_auc',random_state=25)

rf_random.fit(X_train,y_train)
print('mean test scores',rf_random.cv_results_['mean_test_score'])
print('mean train scores',rf_random.cv_results_['mean_train_score'])


# In[128]:


print(rf_random.best_estimator_)


# In[78]:


clf = RandomForestClassifier(bootstrap=True, class_weight={0:1,1:20}, criterion='gini',
            max_depth=14, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=28, min_samples_split=111,
            min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
            oob_score=False, random_state=25, verbose=0, warm_start=False)
clf.fit(X_train , y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[80]:


y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
print("*"*100)
print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[81]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - this model is better from other model in term of AUC , it has also good f1 and recall value but less than Decision tree.

# ## 4.8 XGBOOST Hyperparameter tuning

# In[119]:


param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(110,190),
              "min_samples_leaf": sp_randint(25,65)}

clf = xgb.XGBClassifier(random_state=25,n_jobs=-1)

xgboost = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=10,scoring='roc_auc',random_state=25)

xgboost.fit(X_train,y_train)
print('mean test scores',xgboost.cv_results_['mean_test_score'])
print('mean train scores',xgboost.cv_results_['mean_train_score'])


# In[120]:


print(xgboost.best_estimator_)


# In[141]:


clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=12, min_child_weight=1, min_samples_leaf=33,
       min_samples_split=138, missing=None, n_estimators=109, n_jobs=-1,
       nthread=None, objective='binary:logistic', random_state=25,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[142]:


from sklearn.metrics import roc_curve, auc

clf.fit(X_train , y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[144]:


y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))


# In[147]:


print('Train Detecction rate/recall',recall_score(y_train,y_train_pred))
print('Test Detecction rate/recall',recall_score(y_test,y_test_pred))


# In[148]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation :
# #### 1 -> class 0 and 2 -> class 1
# - the xgboost model is not performed as expected
# - It cant recall well on the test data

# ## 4.9 Basic stacking

# In[39]:


# create base models
# borrowed idea of stacking from AAIC code snippet (from one of the case study)

from sklearn.neighbors import KNeighborsClassifier
# model 1 (KNN)
model_1 = KNeighborsClassifier(n_neighbors =99)
model_1.fit(X_train, y_train)

# model 2 (Decision tree)
model_2 = DTC(class_weight={0:.1,1:15}, criterion='gini', max_depth=100,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=500,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
model_2.fit(X_train, y_train)

# model 3 (Randomforest)
model_3 = RandomForestClassifier(bootstrap=True, class_weight={0:1,1:20}, criterion='gini',
            max_depth=14, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=28, min_samples_split=111,
            min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
            oob_score=False, random_state=25, verbose=0, warm_start=False)
model_3.fit(X_train , y_train)

#model 4 (Xgboost)
model_4 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=12, min_child_weight=1, min_samples_leaf=33,
       min_samples_split=138, missing=None, n_estimators=109, n_jobs=-1,
       nthread=None, objective='binary:logistic', random_state=25,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
model_4.fit(X_train , y_train)


#model 5 (naive bayes)
model_5 =MultinomialNB(alpha = 10)
model_5.fit(X_train,y_train)


# In[43]:


# create a meta classfier (we have logstic regression)
# have not done hyperparam tuning because this code snippet only taking around 3 hour in my box
# so just taking the best param from the previous LR model

meta_clsf = LogisticRegression(C=0.01)
# stack all the 5 model and pass the out put of those model to the meta classfier
stack_clf = StackingClassifier(classifiers = [model_1,model_2,model_3,model_4,model_5], meta_classifier = meta_clsf)
stack_clf.fit(X_train, y_train)


# In[44]:


# get the auc score and plotting
train_fpr, train_tpr, thresholds = roc_curve(y_train, stack_clf.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, stack_clf.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[45]:


y_train_pred = stack_clf.predict(X_train)
y_test_pred = stack_clf.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))

print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[47]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# ### Observation:
# - the above code snippet will take time
# - the above model is normal staking model where i have taken all the model and stacked them
# - pass the output of all model to the meta classfier
# - here also we got a nice auc score but f1 score and recall not at that good

# ##  4.10 Customized Stacking

# #### Method:
# - take the wole data set
# - define train and test data
# - divide the train data into two part: data 1 and data 2 (here we take 50-50 % )
# - create m sample from the train data and fitted with the base learner (xgboost)
# - now predict by passing the data2 to each sample that we are fitted
# - now train the meta classifier with the given predicted value and target value of data 2 (meta clf : logistic regression)
# - now use the fitted meta classifier and predict the test data
#

# In[17]:


# create 2 dataset from train data
data1 = train_data[:65000]
data2 = train_data[65000:]


# In[18]:


# This function is for the preprocessing :- it will craete sample datapoints from the train data
# here is a good source to learn bootstrap sampling : https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/

def preprocessing(train_x ,cv_data, test_data):
    y_train = train_x['label']
    y_cv = cv_data['label']
    y_test = test_data['label']

    # one hot encoding of protocol, service and flag feature
    prototype_vectorizer = CountVectorizer()
    train_protocol_type_encoding = prototype_vectorizer.fit_transform(train_x['protocol_type'])
    cv_protocol_type_encoding = prototype_vectorizer.transform(cv_data['protocol_type'])
    test_protocol_type_encoding = prototype_vectorizer.transform(test_data['protocol_type'])

    service_encode = CountVectorizer()
    train_service_encoding = service_encode.fit_transform(train_x['service'])
    cv_service_encoding = service_encode.transform((cv_data['service']))
    test_service_encoding = service_encode.transform((test_data['service']))

    flag_encoding = CountVectorizer()
    train_flag_encoding = flag_encoding.fit_transform(train_x['flag'])
    cv_flag_encoding = flag_encoding.transform(cv_data['flag'])
    test_flag_encoding = flag_encoding.transform(test_data['flag'])

    # removing correlated features/irrelevant features from train and test data
    X_train_after_FS = train_x.drop(['srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'is_guest_login', 'dst_host_srv_serror_rate', 'num_root', 'dst_host_serror_rate', 'srv_rerror_rate'],axis = 1)
    X_cv_after_FS = cv_data.drop(['srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'is_guest_login', 'dst_host_srv_serror_rate', 'num_root', 'dst_host_serror_rate', 'srv_rerror_rate'],axis = 1)
    X_test_after_FS = test_data.drop(['srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'is_guest_login', 'dst_host_srv_serror_rate', 'num_root', 'dst_host_serror_rate', 'srv_rerror_rate'],axis = 1)


    # removing featues which are adding less value , by looking at the RFECV
    X_train_after_FS = X_train_after_FS.drop(['srv_diff_host_rate', 'num_file_creations' , 'num_failed_logins','num_access_files','root_shell','num_shells','su_attempted','urgent','land', 'num_outbound_cmds','attack'],axis = 1)
    X_cv_after_FS = X_cv_after_FS.drop(['srv_diff_host_rate', 'num_file_creations' , 'num_failed_logins','num_access_files','root_shell','num_shells','su_attempted','urgent','land', 'num_outbound_cmds','attack'],axis = 1) 
    X_test_after_FS = X_test_after_FS.drop(['srv_diff_host_rate', 'num_file_creations' , 'num_failed_logins','num_access_files','root_shell','num_shells','su_attempted','urgent','land', 'num_outbound_cmds','attack'],axis = 1)

    # removing label, attck,protocol_type,service,flag column from train and test data
    X_train_after_FS.drop(['protocol_type','service','flag','label'], axis=1, inplace=True)
    X_cv_after_FS.drop(['protocol_type','service','flag','label'], axis=1, inplace=True)
    X_test_after_FS.drop(['protocol_type','service','flag','label'], axis=1, inplace=True)


    # take protocol_type one_hot_encoding vector and service one_hot_encoding vector and merge them using hstack
    train_protocol_service_encoding = hstack((train_protocol_type_encoding, train_service_encoding))
    cv_protocol_service_encoding = hstack((cv_protocol_type_encoding,  cv_service_encoding))
    test_protocol_service_encoding = hstack((test_protocol_type_encoding,  test_service_encoding))

    # take train_proto_services_encoding vector and flag one_hot_encoding vector and merge them using hstack
    train_protocol_service_flag_encoding = hstack((train_protocol_service_encoding, train_flag_encoding))
    cv_protocol_service_flag_encoding  = hstack((cv_protocol_service_encoding ,cv_flag_encoding))
    test_protocol_service_flag_encoding  = hstack((test_protocol_service_encoding ,test_flag_encoding))


    # merging the categorical onehot encoded feature and numerical feature
    X_train = hstack((train_protocol_service_flag_encoding , X_train_after_FS))
    cv_test = hstack((cv_protocol_service_flag_encoding , X_cv_after_FS))
    X_test = hstack((test_protocol_service_flag_encoding , X_test_after_FS))

    #returning X_train ,y_train, X_test , y_test
    return X_train ,y_train,cv_test,y_cv, X_test , y_test


# In[19]:


X_train ,y_train,cv_test,y_cv, X_test , y_test = preprocessing(data1 ,data2,test_data)


# In[20]:


# print data1 and data2 and test data
print(X_train.shape ,y_train.shape)
print(cv_test.shape ,y_cv.shape)
print(X_test.shape , y_test.shape)


# In[21]:


# this function will shuffle the data and will resample
def shuf_sample(d , y ,num_data_points):
    d , y = shuffle(d, y)
    t_x ,y_x = resample(d ,y, replace=True ,n_samples = num_data_points ,random_state=43)
    return t_x , y_x


# In[22]:


# refer :  https://www.youtube.com/watch?v=enEerl0feRo
# http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/

# this function will take each sample and will going to fit with Xgboost classifier
# also predict the the
def compute_base_learner(train_x,train_y,cv_test,X_test , num_samples ,num_data_points):
    predict_cv = []
    predict_test = []
    for i in range(1,num_samples):
        print("iteration",i)
        train_x , train_y = shuf_sample(X_train ,y_train ,num_data_points)
        clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=12, min_child_weight=1, min_samples_leaf=33,
           min_samples_split=138, missing=None, n_estimators=109, n_jobs=-1,
           nthread=None, objective='binary:logistic', random_state=25,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)
        clf = clf.fit(train_x , train_y)

        pred_cv = clf.predict_proba(cv_test)
        pred_test = clf.predict_proba(X_test)

        predict_cv.append(pred_cv)
        predict_test.append(pred_test)
    return predict_cv , predict_test


# In[23]:


def meta_classifier(predict_cv , predict_test , y_cv):
    prediction_cv = np.column_stack(predict_cv)
    prediction_test = np.column_stack(predict_test)
    meta_clf = LogisticRegression(C = 0.01)
    meta_clf = meta_clf.fit(prediction_cv , y_cv)
    return meta_clf, prediction_test


# ### with 1000 sample

# In[23]:


from datetime import datetime
start = datetime.now()
predict_cv , predict_test = compute_base_learner(X_train ,y_train,cv_test,X_test , 1001 ,1000)
print("Time taken to run this cell :", datetime.now() - start)


# In[25]:


meta_clf ,prediction_test = meta_classifier(predict_cv , predict_test , y_cv)


# In[26]:


# get the auc score and plotting

test_fpr, test_tpr, thresholds = roc_curve(y_test, meta_clf.predict_proba(prediction_test)[:,1])

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")

plt.title("ERROR PLOTS")
plt.show()


# In[27]:


y_test_pred = meta_clf.predict(prediction_test)

print('Test f1 score',f1_score(y_test,y_test_pred))

print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# ### with 1500 sample

# In[28]:


from datetime import datetime
start = datetime.now()
predict_cv , predict_test = compute_base_learner(X_train,y_train,cv_test,X_test , 1501 ,1500)
print("Time taken to run this cell :", datetime.now() - start)


# In[31]:


# get the auc score and plotting
meta_clf ,prediction_test = meta_classifier(predict_cv , predict_test , y_cv)
test_fpr, test_tpr, thresholds = roc_curve(y_test, meta_clf.predict_proba(prediction_test)[:,1])

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")

plt.title("ERROR PLOTS")
plt.show()


# In[32]:


y_test_pred = meta_clf.predict(prediction_test)

print('Test f1 score',f1_score(y_test,y_test_pred))

print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# ### with 2000 sample

# In[34]:


from datetime import datetime
start = datetime.now()
predict_cv , predict_test = compute_base_learner(X_train,y_train,cv_test,X_test , 2001 ,2000)
print("Time taken to run this cell :", datetime.now() - start)


# In[37]:


# get the auc score and plotting
meta_clf ,prediction_test = meta_classifier(predict_cv , predict_test , y_cv)
test_fpr, test_tpr, thresholds = roc_curve(y_test, meta_clf.predict_proba(prediction_test)[:,1])

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")

plt.title("ERROR PLOTS")
plt.show()


# In[38]:


y_test_pred = meta_clf.predict(prediction_test)

print('Test f1 score',f1_score(y_test,y_test_pred))

print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# ### 10000 data point and 100 samples

# In[61]:


from datetime import datetime
start = datetime.now()
predict_cv , predict_test = compute_base_learner(X_train,y_train,cv_test,X_test , 1001 ,10000)
print("Time taken to run this cell :", datetime.now() - start)


# In[69]:


# get the auc score and plotting
meta_clf , prediction_test = meta_classifier(predict_cv , predict_test , y_cv)


# In[71]:


test_fpr, test_tpr, thresholds = roc_curve(y_test, meta_clf.predict_proba(prediction_test)[:,1])

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")

plt.title("ERROR PLOTS")
plt.show()


# In[72]:


y_test_pred = meta_clf.predict(prediction_test)

print('Test f1 score',f1_score(y_test,y_test_pred))

print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# ## 4.11 Feature Engineering
# lets add two new feature
# - add 2 most important feature as per feature selection
# - use square of an important feature

# In[17]:


train_data['same_srv_rate_src_bytes'] = train_data['same_srv_rate']+ train_data['src_bytes']
test_data['same_srv_rate_src_bytes'] =test_data['same_srv_rate']+ test_data['src_bytes']


# In[18]:


train_data['same_srv_rate_sqr'] = train_data['same_srv_rate'] ** 2
test_data['same_srv_rate_sqr'] =test_data['same_srv_rate'] ** 2


# - Please run the cell after feature selection section upto standardization

# In[50]:


train_data.head()


# In[32]:


param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(110,190),
              "min_samples_leaf": sp_randint(25,65)}

clf = RandomForestClassifier(random_state=25,n_jobs=-1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=10,scoring='roc_auc',random_state=25)

rf_random.fit(X_train,y_train)
print('mean test scores',rf_random.cv_results_['mean_test_score'])
print('mean train scores',rf_random.cv_results_['mean_train_score'])


# In[33]:


print(rf_random.best_estimator_)


# In[36]:


clf = RandomForestClassifier(bootstrap=True, class_weight={0:1,1:20}, criterion='gini',
            max_depth=14, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=28, min_samples_split=111,
            min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
            oob_score=False, random_state=25, verbose=0, warm_start=False)
clf.fit(X_train , y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[38]:


y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
print("*"*100)
print("train recall score / detection rate",recall_score(y_train,y_train_pred))
print("test recall score / detection rate",recall_score(y_test,y_test_pred))


# In[41]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)


# #### Observation:
#     - this model is giving highest auc value better than our previous random forest model

# ### Understanding the whole model building section breifly
# - Merge all the encoded categorical and numerical features
# - build a base line model in this case we have taken Naive bayes (we can take knn also)
# - By looking at the naive bayes model after hyperparameter tuning the model seems to overfit(gap in train and test score)
# - to overcome from this problem we select features by recusive feature elemination (please read the "understanding feature selection" after 4.2 section)
# - after done that remove all the feature manually from the train data and those features which are not adding value by seeing the important feature plot
# - Build the model again starting with the naive bayes model
# - after feature selection we got good AUC but there were some problem like f1 score was low and confusion on the test recall , this was beacause the data set is imbalanced , so give some class weight to tackle this , and after that we got pretty good result.
# - So far there are two good model Decision tree and Randomforest
# - I thought Xgboost will gonna give best result but thats ok .
# - tried two kind of stacking but the result were not at all good in term of f1 score and recall score.
# - In the feature engineering section i have tried 2 new feature which is take the those feature which are important by our feature selection method , then add those feature and other is square a feature 
# - The result of feature engineered model(trioed random forest) is great this is highest auc value i have got , but the f1 and recall score is lower than Decision tree and random forest

# In[49]:


from prettytable import PrettyTable
k = PrettyTable()
p = PrettyTable()
print('*************Before feature selection**************')
k.field_names = ["Model","Train AUC" ,"Test AUC" ,"f1 score on test data" , "recall on test data"]
k.add_row(["Naive Bayes" , .9829 , 0.8400,0.7687,.6388])
print(k)
print('*********After feature selection**************')
p.field_names = ["Model","Train AUC" ,"Test AUC" ,"f1 score on test data" , "recall on test data"]
p.add_row(["Naive Bayes" , .9863 , 0.8553,0.7734,.6452])
p.add_row(["KNN" , .999 , 0.891,0.7806,.6593])
p.add_row(["Logistic Regression" , .9928 , 0.9049,0.7264,.6032])
p.add_row(["Decision Tree" , .9998 , 0.9011,0.8820,.8549])
p.add_row(["Random Forest" , .9999 , 0.9684,0.8754,.8030])
p.add_row(["Xgboost" , .9999 , 0.9672,0.7592,.6253])
p.add_row(["Basic stacking model" , .9999 , 0.9614,0.7843,.6590])
p.add_row(["Customized stacking model" , "not computed" , 0.957,0.79,.666])
p.add_row([" RF with feature engineering" , .999 , .972 , .873, .801])
print(p)

