import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

# Location of dataset
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
url2 = "https://www.openml.org/data/get_csv/44/dataset_44_spambase.arff"

# Read dataset to pandas dataframe
letterdata_origin = pd.read_csv(url1, header= None)
spamdata_origin = pd.read_csv(url2) 

def sample_preprocess_letter(frac_now):
    # Sampling
    print('data percentage: ' + str(frac_now))
    letterdata = letterdata_origin.sample(frac=frac_now)
    print('data shape: ' + str(letterdata.shape))

    # Preprocess
    X = letterdata.drop(0, axis=1).astype(float)
    y = letterdata[0]
    # le = preprocessing.LabelEncoder()
    # y = y.apply(le.fit_transform)  
    y = y.map(ord) - ord('A') + 1
    return X, y

def sample_preprocess_spam(frac_now):
    # Sampling
    print('data percentage: ' + str(frac_now))
    spamdata = spamdata_origin.sample(frac=frac_now)
    print('data shape: ' + str(spamdata.shape))

    # Preprocess
    X = spamdata.drop('class', axis=1).astype(float)
    y = spamdata['class']
    # le = preprocessing.LabelEncoder()
    # y = y.apply(le.fit_transform)  
    return X, y

def accuracy_over_depth(dataset_name):
  scores_train = []
  scores_test = []
  scores_cv10 = []
  depths =[1,5,9,13,17,21,25,29,33,37,41]

  for depth_now in depths:
    print(depth_now)
    
    if dataset_name == 'letter':
      X, y = sample_preprocess_letter(0.2)
    elif dataset_name == 'spam':
      X, y = sample_preprocess_spam(0.2)
    else:
      return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # Standardlization
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    # Train Model
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth_now),
                           n_estimators=50, learning_rate=1)
    boost.fit(X_train, y_train)

    # Cross Validation
    boost_cv = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth_now),
                           n_estimators=50, learning_rate=1)
    pipeline = Pipeline([('transformer', scaler), ('estimator', boost_cv)])
    cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

    # Save Scores
    scores_train.append(boost.score(X_train, y_train))
    scores_test.append(boost.score(X_test, y_test))
    scores_cv10.append(cv_scores.mean())

  return scores_train, scores_test, scores_cv10, depths

def accuracy_over_iteration(dataset_name):
  scores_train = []
  scores_test = []
  scores_cv10 = []
  iters =[1,2,3,4,5,6,7,8,9,10,30,50,70,90,110,130,150,170,190]

  for iter_now in iters:
    print(iter_now)
    
    if dataset_name == 'letter':
      X, y = sample_preprocess_letter(0.2)
    elif dataset_name == 'spam':
      X, y = sample_preprocess_spam(0.2)
    else:
      return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # Standardlization
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    # Train Model
    boost = AdaBoostClassifier(DecisionTreeClassifier(),
                           n_estimators=iter_now, learning_rate=1)
    boost.fit(X_train, y_train)

    # Cross Validation
    boost_cv = AdaBoostClassifier(DecisionTreeClassifier(),
                           n_estimators=iter_now, learning_rate=1)
    pipeline = Pipeline([('transformer', scaler), ('estimator', boost_cv)])
    cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

    # Save Scores
    scores_train.append(boost.score(X_train, y_train))
    scores_test.append(boost.score(X_test, y_test))
    scores_cv10.append(cv_scores.mean())

  return scores_train, scores_test, scores_cv10, iters

def accuracy_over_datasize(dataset_name):
  scores_train = []
  scores_test = []
  scores_cv10 = []
  fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  
  for frac_now in fracs:
    print(frac_now)
    
    if dataset_name == 'letter':
      X, y = sample_preprocess_letter(frac_now=frac_now)
    elif dataset_name == 'spam':
      X, y = sample_preprocess_spam(frac_now=frac_now)
    else:
      break
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # Standardlization
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    # Train Model
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=17),
                           n_estimators=90, learning_rate=1)
    boost.fit(X_train, y_train)

    # Cross Validation
    boost_cv = AdaBoostClassifier(DecisionTreeClassifier(max_depth=17),
                           n_estimators=90, learning_rate=1)
    pipeline = Pipeline([('transformer', scaler), ('estimator', boost_cv)])
    cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

    # Save Scores
    scores_train.append(boost.score(X_train, y_train))
    scores_test.append(boost.score(X_test, y_test))
    scores_cv10.append(cv_scores.mean())
   
  return scores_train, scores_test, scores_cv10, fracs

def plot(scores_train, scores_test, scores_cv10, name, array, title):
  print(np.average(scores_train), np.average(scores_test), np.average(scores_cv10))
  print(scores_train)
  print(scores_test)
  print(scores_cv10)

  plt.plot(scores_train, color='green', alpha=0.8, label='Train')
  plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
  plt.plot(scores_cv10, color='blue', alpha=0.8, label='CV-10')
  plt.title("Accuracy over " + title + "(" + name + ")", fontsize=14)
  plt.ylabel('Accuracy')
  plt.xlabel(title)
  plt.xticks(np.arange(len(array)), array)
  plt.legend(loc='best')
  dwn = plt.gcf()
  plt.show()

# Dataset1 train and test
# Accuracy over Max Tree Depth
scores_train, scores_test, scores_cv10, array = accuracy_over_depth(dataset_name='letter')
plot(scores_train, scores_test, scores_cv10, 'letter', array, 'Max Tree Depth')
​
# Accuracy over Max Iterations
scores_train, scores_test, scores_cv10, array = accuracy_over_iteration(dataset_name='letter')
plot(scores_train, scores_test, scores_cv10, 'letter', array, 'Max Iterations')
​
# Accuracy over Datasize Percentage
scores_train, scores_test, scores_cv10, array = accuracy_over_datasize(dataset_name='letter')
plot(scores_train, scores_test, scores_cv10, 'letter', array, 'Datasize Percentage')
​
# Dataset2 train and test
# Accuracy over Max Tree Depth
# scores_train, scores_test, scores_cv10, array = accuracy_over_depth(dataset_name='spam')
# plot(scores_train, scores_test, scores_cv10, 'spam', array, 'Max Tree Depth')
​
# Accuracy over Max Iterations
# scores_train, scores_test, scores_cv10, array = accuracy_over_iteration(dataset_name='spam')
# plot(scores_train, scores_test, scores_cv10, 'spam', array, 'Max Iterations')
​
# Accuracy over Datasize Percentage
# scores_train, scores_test, scores_cv10, array = accuracy_over_datasize(dataset_name='spam')
# plot(scores_train, scores_test, scores_cv10, 'spam', array, 'Datasize Percentage')