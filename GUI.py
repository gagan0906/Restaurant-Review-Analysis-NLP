from PyQt5.QtWidgets import QMainWindow, QApplication, QPlainTextEdit, QLabel, QPushButton
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
with open('model.pickle','rb') as f:
      model = pickle.load(f)
'''
with open('countVector.pickle','rb') as f:
      cv = pickle.load(f)
      
with open('tfidf.pickle','rb') as f:
      tfidf = pickle.load(f)      
'''

with open('tfidfCV.pickle','rb') as f:
      tfidf = pickle.load(f)      
        


class window(QMainWindow):
      def __init__(self):
            super().__init__()
            self.title = "Restaurant Reviews"
            self.review = 'Food is not very good here'
            self.initialize()
            self.input()
            self.show()

      def initialize(self):
            self.setFixedSize(600,400)
            self.setWindowTitle = self.title
            label = QLabel("Write your review",self)
            label.move(20,10)

            calculateReview = QPushButton("Compute Review",self)
            calculateReview.move(500,300)
            calculateReview.clicked.connect(self.process)

            
      def input(self):
            text = QPlainTextEdit(self)
            text.move(20,40)
            text.resize(400,200)  
            #self.review = QPlainTextEdit.toPlainText
             

      def process(self):
            review = re.sub('[^a-zA-Z]',' ',self.review)
            lemmatizer = WordNetLemmatizer()

            review = [lemmatizer.lemmatize(word) for word in self.review]
            review = ''.join(review)
           
            li = [review] 
            print(type(li))
            
            
            #cv= CountVectorizer(max_features=1200)
            
                        
            
            res = tfidf.transform(li)
            res = model.predict(res)
            print(res)
            


qt = QApplication(sys.argv)
w = window()
sys.exit(qt.exec())            