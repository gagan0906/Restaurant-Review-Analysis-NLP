import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t')

corpus = []

for i in range(0,1000):
      review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
      
      review = review.lower()
      review = review.split()  
      
    
      lemmatizer = WordNetLemmatizer()
  
       
      review = [lemmatizer.lemmatize(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
      review = ' '.join(review)   
      corpus.append(review)  

'''
vectorizer = CountVectorizer(max_features=1200)
X= vectorizer.fit_transform(corpus).toarray()
'''

tfidf = TfidfVectorizer(max_features=1200)
X = tfidf.fit_transform(corpus).toarray()

'''
tfidf = TfidfTransformer()
X = tfidf.fit_transform(X)
'''

y= df.iloc[:,1].values

from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=510)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)


#from sklearn.metrics import confusion_matrix
#ac = confusion_matrix(y_test,pred)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,pred)

import pickle
with open('model.pickle','wb') as f:
      pickle.dump(classifier,f)
      
with open('tfidfCV.pickle','wb') as f:
      pickle.dump(tfidf,f)
      
#with open('countVector.pickle','wb') as f:
#      pickle.dump(CountVectorizer,f)
      
#with open('tfidf.pickle','wb') as f:
#      pickle.dump(tfidf,f)
      

