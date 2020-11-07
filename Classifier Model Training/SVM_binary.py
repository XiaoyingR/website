import re
import string
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import nltk
from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.porter import *
# from nltk.corpus import stopwords
import numpy as np

stemmer = SnowballStemmer("english")
# stemmer = PorterStemmer()
# stop_list = stopwords.words('english')

import pickle

cat1_file = open("NEALTA_pos.txt", "r", encoding="utf8")
cat2_file = open("NEALTA_neg.txt", "r", encoding="utf8")


def reading(in_file):
    rawcorpus = []
    # .readlines() to read the whole file into memory and returns its contents as a list of its lines
    for line in in_file.readlines():
        # .strip() to remove whitespace from start and end
        # .append to add line to list
        rawcorpus.append(line.strip())
    return rawcorpus


def preprocessing_removeEmpty(corpus, label):
    remove = string.punctuation
    pattern = r"[{}]".format(remove)
    rawCorpus = []
    processedCorpus = []
    processedLabel = []
    print('Original size: ' + str(len(corpus)))
    for i in range(len(corpus)):
        match = re.search(r'^[\w\.\d\_\-]+', corpus[i])
        username = match.group()

        # Split each line into sentences
        sents = nltk.sent_tokenize(corpus[i])
        # print ('This is sents' + str(i) + ': ' + str(sents))
        alllines = []
        # ngrams = []

        # For Sentence in line
        for sent in sents:
            line = sent
            line = re.sub("https:\/\/t\.co\/\S+", "wURL", line)
            line = re.sub(username, "", line)
            line = re.sub("#", "wHashtag ", line)
            line = re.sub("@", "wAtUser ", line)
            # line = re.sub("\s+"," ",line)

            # Searches for all the instances of pattern in the line, and replaces them with "".
            line = re.sub(pattern, "", line)

            # line = re.sub("[\d\:]+[aA][mM]","am",line)
            # line = re.sub("[\d\:]+[pP][mM]","pm",line)
            # line = re.sub("(ha)+","ha",line)
            # line = re.sub("a(h)+","ahh",line)

            # .split(): by default - split by space
            doc1 = line.split()
            doc2 = [w for w in doc1 if re.search('^[a-zA-Z]+$', w)]
            doc3 = [w.lower() for w in doc2]
            doc4 = [w for w in doc3 if len(w) >= 2]
            doc5 = [stemmer.stem(w) for w in doc4]
            for w in doc5:
                alllines.append(w)

                # for w in doc5:
                # 	ngrams.append(w)

                # #create bigram/trigram to make it read as a phrase
                # if len(doc5)>=2:
                # 	for j in range(len(doc5)-1):
                # 		bigram = doc5[j]+"A"+doc5[j+1]
                # 		ngrams.append(bigram)
                # if len(doc5)>=3:
                # 	for k in range(len(doc5)-2):
                # 		trigram = doc5[k]+"A"+doc5[k+1]+"A"+doc5[k+2]
                # 		ngrams.append(trigram)

        # joins all the phrases/words in the list with a space in between each phrase/word
        # newline = " ".join(ngrams)
        newline = " ".join(alllines)
        newline = username + " " + newline

        # if newLine is not empty
        if len(newline) >= 1:
            processedCorpus.append(newline)
            rawCorpus.append(corpus[i])
            processedLabel.append(label[i])
    return (processedCorpus, rawCorpus, processedLabel)


print('Reading files ... ')
# to obtain text in file as list of lines (whitespace removed)
# corpus_cat1 (pos) & corpus_cat2 (neg) are sample files - for training
corpus_cat1 = reading(cat1_file)
corpus_cat2 = reading(cat2_file)
# both files corpus combined
corpus = corpus_cat1 + corpus_cat2
print('size of corpus: ' + str(len(corpus)))

# repeat the tags
# corpus_cat1 consists of positive training data
y = ['pos'] * len(corpus_cat1)
# corpus_cat2 consists of negative training data
y.extend(['neg'] * (len(corpus_cat2)))

raw_train, raw_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=5)

# to balance positive & negative training data 
for i in range(len(raw_train)):
    if y_train[i] == 'pos':
        for j in range(3):  # multiply positive data by 3 times
            raw_train.append(raw_train[i])
            y_train.append(y_train[i])
            j += 1

# print(len(raw_train))

# tuple of processed corpus, raw corpus, processed label
# For training data, must remove the empty lines to ensure accurate training
(X_train, raw_train, y_train) = preprocessing_removeEmpty(raw_train, y_train)
print('Number of lines (original tweets training): ' + str(len(X_train)))
# print(len(y_train))
# print(len(raw_test))

# tuple of #tuple of processed corpus (test), raw corpus(test), processed label (test)
# .. Note: Not actual test data, just randomized test created from initial corpus (see line 187)
# For testing data, can leave the empty lines in to check if the classifier is precise.
(X_test, raw_test, y_test) = preprocessing_removeEmpty(raw_test, y_test)
print('Number of lines (original tweets testing): ' + str(len(X_test)))

unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
temp_uni_tfidf = unigram_vectorizer.fit_transform(X_train).toarray()
# n_features = len(unigram_vectorizer.get_feature_names())
n_features = 9000
multigrams_vectorizer = TfidfVectorizer(ngram_range=(2, 2), min_df=2, max_features=n_features)
comb_vectorizer = FeatureUnion([("uni_vec", unigram_vectorizer), ("multi_vec", multigrams_vectorizer)])
# comb_vectorizer.set_params(multi_vec=None)
X_tfidf = comb_vectorizer.fit_transform(X_train).toarray()
with open("vectorizer_NEALTA_Binary.pk", "wb") as vect_file:
    pickle.dump(comb_vectorizer, vect_file)

feature_names = comb_vectorizer.get_feature_names()
print("num_features: " + str(len(feature_names)))
# print(feature_names[:50])
print("features extracted & tfidf transformed")
# Transform documents to document-term matrix. (.transform) - No learning involved as it is test data 
# For test data 
# X_test_tfidf = vectorizer.transform(X_test).toarray()
X_test_tfidf = comb_vectorizer.transform(X_test).toarray()

print('Creating Linear SVC Model...')
# model=svm.LinearSVC(C=1000)
model = svm.SVC(kernel="linear", C=1000, cache_size=5000, probability=True)
# model = svm.SVC(kernel="rbf", C=1, cache_size=5000,probability=True)

print('Linear SVC Model created!')
# .fit(X, y[, sample_weight]): Fit the model according to the given training data
# For training data 
# parameters = [{'C': [1, 10, 100, 1000]}]
# parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
# clf = GridSearchCV(model, parameters,return_train_score=True)

print('Training in progress...')
model.fit(X_tfidf, y_train)
# clf.fit(X_tfidf,y_train)
print('Training completed!')
# .score(X, y[, sample_weight]): Returns the mean accuracy on the given test data and labels
# For training data 
print("Training score -- " + str(model.score(X_tfidf, y_train)))
with open("clf_NEALTA_Binary.pk", "wb") as clf_file:
    pickle.dump(model, clf_file)
# print("Training score -- " +str(clf.score(X_tfidf,y_train)))
# with open("clf_pestcontrol.pk","wb") as clf_file:
#                 pickle.dump(clf,clf_file)


# .predict(X): Predict classs labels for samples in X 
# For test data 
y_predicted = model.predict(X_test_tfidf)
# y_predicted = clf.predict(X_test_tfidf)
# predict_prob = model.predict_proba(X_test_tfidf)

print("the following tweets are predicted to be relevant:")
print('Predicted' + '  ' + 'Test' + '  ' + 'Tweet')
print('-----------------------------------------')

positiveOutput = open('NEALTA_binary_finalpositive.txt', 'w', encoding="utf8")
'''
for i in range(len(y_predicted)):
	if y_predicted[i] == 'pos':
		print(str(y_predicted[i])+"\t"+str(y_test[i])+"\t"+raw_test[i])
		positiveOutput.write(raw_test[i]+'\n')
'''
totalp = 0
p = 0
totalr = 0
r = 0
for i in range(len(y_predicted)):
    if y_predicted[i] == 'pos':
        # if positive, add to totalp count
        totalp += 1
        if y_predicted[i] == y_test[i]:
            # if prediction is correct, add to p count
            p += 1
    if y_test[i] == 'pos':
        totalr += 1
        if y_predicted[i] == y_test[i]:
            r += 1
            # print(str(i+1)+": predicted to be: "+str(y_predicted[i])+" labelled as: "+str(y_test[i]))

prec = float(p) / totalp
recal = float(r) / totalr
# formula for F1 Score
F = 2 * prec * recal / (prec + recal)

# round(number [, ndigits]): Returning the floating point value number rounded to ndigits digits after the decimal point 
# .. if ndigits is omitted, default to 0
prec = round(prec, 2)
recal = round(recal, 2)
F = round(F, 2)
print()
print()
# np.mean( ): np - numpy 
# .. Compute the arithmetic mean along the specified axis.
print("Testing score -- " + str(np.mean(y_predicted == y_test)))
print("Precision -- " + str(prec))
print("Recall -- " + str(recal))
print("F_score -- " + str(F))
