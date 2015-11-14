import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

with open('SMSSpamCollection') as file:
    dataset = [[x.split('\t')[0],x.split('\t')[1]] for x in [line.strip() for line in file]]


data = [dat[1] for dat in dataset]
labels = [dat[0] for dat in dataset]

#print labels[:5]
#print data[:5]

pipeline = Pipeline ([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(data[:3000],labels[:3000])

predictions = pipeline.predict(data[3000:])

score = 0

for i in range(3000,len(data)):
    if predictions[i-3000] == labels[i]:
        score = score + 1
        #print score

print score*1.0/len(data[3000:])

