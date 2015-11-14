import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer



with open('SMSSpamCollection') as file:
    dataset = [[x.split('\t')[0],x.split('\t')[1]] for x in [line.strip() for line in file]]


data = np.array([dat[1] for dat in dataset])
labels = np.array([dat[0] for dat in dataset])

#print labels[:5]
#print data[:5]

pipeline = Pipeline ([
#    ('vectorizer', CountVectorizer(ngram_range=(1,2))),
#    ('tfidf_transformer', TfidfTransformer()),
#    ('extract_essays', EssayExractor()),
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
          ('counts', CountVectorizer()),
#          ('tf_idf', TfidfTransformer())
          ])),
        #('essay_length', LengthTransformer()),
        #('misspellings', MispellingCountTransformer())
    ])),
    ('classifier', MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=10)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    #print train_indices
    #print test_indices
    train_text = data[train_indices]
    train_y = labels[train_indices]

    test_text = data[test_indices]
    test_y = labels[test_indices]

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

#pipeline.fit(data[:3000],labels[:3000])

#predictions = pipeline.predict(data[3000:])

#score = 0

#for i in range(3000,len(data)):
#    if predictions[i-3000] == labels[i]:
#        score = score + 1
        #print score

#print score*1.0/len(data[3000:])

