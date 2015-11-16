import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin
from pandas import DataFrame
import matplotlib.pyplot as plt

#Create custom transformer classes to use in analysis

#UpperCaseTransformer used to find % of text in message that is uppercase
class UpperCaseTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return DataFrame([sum(1 for c in string if c.isupper())*1.0/len(string) for string in X])

    def fit(self, X, y=None, **fit_params):
        return self
        
#EllipsesTransformer used to determine if text contains instance of "..."
class EllipsesTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return DataFrame([ 1 if string.find("...") != -1 else 0 for string in X])

    def fit(self, X, y=None, **fit_params):
        return self
      
#Function for performing an analysis based on the different       
def performAnalysis(pipeline):
    #use K Folds for cross-validation to perform analysis on unseen data
    k_fold = KFold(n=len(data), n_folds=10)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data[train_indices]
        train_y = labels[train_indices]

        test_text = data[test_indices]
        test_y = labels[test_indices]

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
        
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='spam')
        scores.append(score)

    #print out reporting data
    print('Total emails classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)

#Read all the data from the file
with open('SMSSpamCollection') as file:
    dataset = [[x.split('\t')[0],x.split('\t')[1]] for x in [line.strip() for line in file]]

#Extract out the data and labels (ham,spam) from the dataset file
data   = np.array([dat[1] for dat in dataset])
labels = np.array([dat[0] for dat in dataset])


#Create all of the pipelines that will connect a series of steps to execute
testPipeline = Pipeline ([
#    ('vectorizer', CountVectorizer(ngram_range=(1,2))),
#    ('tfidf_transformer', TfidfTransformer()),
#    ('extract_essays', EssayExractor()),
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
          ('counts', CountVectorizer()),
#            ('tf_idf', TfidfTransformer())
          ])),
        ('percentage_uppercase', UpperCaseTransformer()),
#        ('ellipses', EllipsesTransformer())
        #('essay_length', LengthTransformer()),
        #('misspellings', MispellingCountTransformer())
    ])),
    ('classifier', MultinomialNB())
])

#Simple NB pipeline example to give us a baseline for comparison
basicNBPipeline = Pipeline ([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
          ('counts', CountVectorizer()),
          ])),
    ])),
    ('classifier', MultinomialNB())
])

#Execute the test cases
performAnalysis(testPipeline)

u = UpperCaseTransformer()
e = EllipsesTransformer()
ellipsescounts = e.transform(data)
ellipsespam = 0
ellipseham = 0
print ellipsescounts.iat[3,0]
for ellipse in range(len(ellipsescounts)):
    if ellipsescounts.iat[ellipse,0] == 1:
        if labels[ellipse] == 'ham':
            ellipseham = ellipseham+1
        else:
            ellipsespam = ellipsespam + 1
            
uppercasecounts = u.transform(data)
#print uppercasecounts[:10]
labelsBinary = []
for label in labels:
    if label == 'ham':
        labelsBinary.append(1)
    else:
        labelsBinary.append(0)

plt.subplot(2,1,1)
plt.scatter(labelsBinary, 100*uppercasecounts)
plt.xlabel('Labels 1:ham, 0:spam')
plt.ylabel('% capitalization')

plt.subplot(2,1,2)
plt.scatter([0, 1], [ellipsespam, ellipseham])
plt.xlabel('Labels 1:ham, 0:spam')
plt.ylabel('# of ellipses')
plt.show()


