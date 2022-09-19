
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

df = pd.read_csv('/Users/abubakri/Library/Containers/com.microsoft.Excel/Data/Downloads/fake-news.csv')




df1 = df.copy()
df1['Real'] = df1['Real'].astype(str).map(lambda x: x.replace('0', 'Fake'))
df1['Real'] = df1['Real'].astype(str).map(lambda x: x.replace('1', 'Real'))




X = df['Content']
y = df['Real']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=7, stratify = y)

tfidf_vectorizer=TfidfVectorizer(stop_words='english')

#DataFlair - Fit and transform train set, transform test set
X_train=tfidf_vectorizer.fit_transform(X_train) 
X_test=tfidf_vectorizer.transform(X_test)




DT = DecisionTreeClassifier(random_state = 42) # initialising KNeighbors Classifier
NB = GaussianNB()# initialising Naive Bayes
RF = RandomForestClassifier(random_state = 42)



lr = LogisticRegression() # defining meta-classifier
clf_stack = StackingClassifier(classifiers =[NB, DT, RF], meta_classifier = lr, use_probas = True, use_features_in_secondary = True)


model_stack = clf_stack.fit(X_train.toarray(), y_train) # training of stacked model
pred_stack = model_stack.predict(X_test.toarray())	 # predictions on test data using stacked model


news = '19 bombed in kaduna train'


pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))




