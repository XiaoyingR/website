import re
import string
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
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

est1_file = open('NUNLTA_1eSTL.txt', 'r', encoding="utf8")
pt2_file = open('NUNLTA_2PT.txt', 'r', encoding="utf8")
enf3_file = open('NUNLTA_3Enf.txt', 'r', encoding="utf8")
enf4_file = open('NUNLTA_4Enf.txt', 'r', encoding="utf8")
others5_file = open('NUNLTA_5Others.txt', 'r', encoding="utf8")
clean6_file = open('NUNLTA_6Clean.txt', 'r', encoding="utf8")
flood7_file = open('NUNLTA_7Flood.txt', 'r', encoding="utf8")
noise8_file = open('NUNLTA_8Noise.txt', 'r', encoding="utf8")
traffic9_file = open('NUNLTA_9Traffic.txt', 'r', encoding="utf8")
pubsafety10_file = open('NUNLTA_10PubSafety.txt', 'r', encoding="utf8")
env11_file = open('NUNLTA_11Env.txt', 'r', encoding="utf8")
infrastructure12_file = open('NUNLTA_12Infrastructure.txt', 'r', encoding="utf8")
est13_file = open('NUNLTA_13eSTNUN.txt', 'r', encoding="utf8")
enf14_file = open('NUNLTA_14Enf.txt', 'r', encoding="utf8")
pollhaze15_file = open('NUNLTA_15PollHaze.txt', 'r', encoding="utf8")
poll16_file = open('NUNLTA_16Poll.txt', 'r', encoding="utf8")
food17_file = open('NUNLTA_17Food.txt', 'r', encoding="utf8")
pestc18_file = open('NUNLTA_18PestC.txt', 'r', encoding="utf8")
animal19_file = open('NUNLTA_19Animal.txt', 'r', encoding="utf8")


# others20_file = open('NEALTA_20Others.txt','r', encoding = "utf8")
# compliment21_file = open('NEALTA_21Compliment.txt','r', encoding = "utf8")
# negfile_file = open('NEALTA_neg.txt','r', encoding = "utf8")


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
            # line = re.sub("#","wHashtag ",line)
            # line = re.sub("@","wAtUser ",line)
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
    return processedCorpus, rawCorpus, processedLabel


print('Reading files ... ')
# to obtain text in file as list of lines (whitespace removed)
# corpus_cat1 (pos) & corpus_cat2 (neg) are sample files - for training
corpus_cat1 = reading(est1_file)
corpus_cat2 = reading(pt2_file)
corpus_cat3 = reading(enf3_file)
corpus_cat4 = reading(enf4_file)
corpus_cat5 = reading(complain5_file)
corpus_cat6 = reading(clean6_file)
corpus_cat7 = reading(flood7_file)
corpus_cat8 = reading(noise8_file)
corpus_cat9 = reading(traffic9_file)
corpus_cat10 = reading(pubsafety10_file)
corpus_cat11 = reading(env11_file)
corpus_cat12 = reading(infrastructure12_file)
corpus_cat13 = reading(est13_file)
corpus_cat14 = reading(enf14_file)
corpus_cat15 = reading(pollhaze15_file)
corpus_cat16 = reading(poll16_file)
corpus_cat17 = reading(food17_file)
corpus_cat18 = reading(pestc18_file)
corpus_cat19 = reading(animal19_file)
corpus_cat20 = reading(others20_file)
corpus_cat21 = reading(compliment21_file)
# corpus_cat22 = reading(negfile_file)

# both files corpus combined
corpus = corpus_cat1 + corpus_cat2 + corpus_cat3 + corpus_cat4 + corpus_cat5 + corpus_cat6 + corpus_cat7 + corpus_cat8 + corpus_cat9 + corpus_cat10 + corpus_cat11 + corpus_cat12 + corpus_cat13 + corpus_cat14 + corpus_cat15 + corpus_cat16 + corpus_cat17 + corpus_cat18 + corpus_cat19 + corpus_cat20 + corpus_cat21
print('size of corpus: ' + str(len(corpus)))

# repeat the tags
y = ['eServices_LTA'] * len(corpus_cat1)
y.extend(['Public_Transport'] * (len(corpus_cat2)))
y.extend(['Enforcement_Cycling'] * (len(corpus_cat3)))
y.extend(['Enforcement_Vehicle'] * (len(corpus_cat4)))
y.extend(['Compliants_NEA/LTA'] * (len(corpus_cat5)))
y.extend(['Cleanliness'] * (len(corpus_cat6)))
y.extend(['Flooding'] * (len(corpus_cat7)))
y.extend(['Noise'] * (len(corpus_cat8)))
y.extend(['Traffic_Congestion'] * (len(corpus_cat9)))
y.extend(['Public_Safety'] * (len(corpus_cat10)))
y.extend(['Enviroment'] * (len(corpus_cat11)))
y.extend(['Infrastructure'] * (len(corpus_cat12)))
y.extend(['eServices_NEA'] * (len(corpus_cat13)))
y.extend(['Enforcement_Human_Acticities'] * (len(corpus_cat14)))
y.extend(['Pollution_Haze'] * (len(corpus_cat15)))
y.extend(['Pollution'] * (len(corpus_cat16)))
y.extend(['Food'] * (len(corpus_cat17)))
y.extend(['Pest_Control'] * (len(corpus_cat18)))
y.extend(['Animal_Issues'] * (len(corpus_cat19)))
y.extend(['Others/Enquiries/Appeals'] * (len(corpus_cat20)))
y.extend(['Compliment_NEA/LTA'] * (len(corpus_cat21)))
# y.extend(['neg']*(len(corpus_cat22)))

raw_train, raw_test, y_train, y_test = train_test_split(corpus, y, test_size=0.05, random_state=5)

'''
# to balance positive & negative training data 
for i in range(len(raw_train)):
	if y_train[i] != 'neg':
		for j in range(1): # multiply positive data by 2 times 
			raw_train.append(raw_train[i])
			y_train.append(y_train[i])
			j+=1
'''
# print(len(raw_train))

# tuple of processed corpus, raw corpus, processed label
# For training data, must remove the empty lines to ensure accurate training
(X_train, raw_train, y_train) = preprocessing_removeEmpty(raw_train, y_train)
print('Number of lines training set: ' + str(len(X_train)))
(X_test, raw_test, y_test) = preprocessing_removeEmpty(raw_test, y_test)
print('Number of lines testing set: ' + str(len(X_test)))

unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
temp_uni_tfidf = unigram_vectorizer.fit_transform(X_train).toarray()
# n_features = len(unigram_vectorizer.get_feature_names())
n_features = 100
multigrams_vectorizer = TfidfVectorizer(ngram_range=(2, 3), min_df=2, max_features=n_features)
comb_vectorizer = FeatureUnion([("uni_vec", unigram_vectorizer), ("multi_vec", multigrams_vectorizer)])
comb_vectorizer.set_params(multi_vec=None)
X_train_tfidf = comb_vectorizer.fit_transform(X_train).toarray()

feature_names = comb_vectorizer.get_feature_names()
print("num_features: " + str(len(feature_names)))
# print(feature_names[:50])
print("features extracted & tfidf transformed.")
# Transform documents to document-term matrix. (.transform) - No learning involved as it is test data 
# For test data 
# X_test_tfidf = vectorizer.transform(X_test).toarray()
X_test_tfidf = comb_vectorizer.transform(X_test).toarray()

# model = svm.SVC(kernel="linear",C=100,cache_size=5000,probability=True)
print('Creating Model...')
model = svm.LinearSVC(C=10, multi_class='ovr')
# model = svm.SVC(kernel="linear",C=10,decision_function_shape='ovr',cache_size=5000,probability=True)
# model = svm.SVC(kernel="rbf",C=10000,decision_function_shape='ovr',cache_size=5000,probability=True)

print('Model created!')
# .fit(X, y[, sample_weight]): Fit the model according to the given training data
# For training data 
# parameters = [{'C': [1, 10, 100, 1000]}]
# parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
# clf = GridSearchCV(model, parameters,return_train_score=True)

print('Training in progress...')
model.fit(X_train_tfidf, y_train)
# clf.fit(X_train_tfidf,y_train)
print('Training completed!')
# For training data
print("Training score -- " + str(model.score(X_train_tfidf, y_train)))

# dump trained vectorizer and model
with open("vectorizer_NEALTA_multi_v2.pk", "wb") as vect_file:
    pickle.dump(comb_vectorizer, vect_file)
with open("clf_NEALTA_multi_v2.pk", "wb") as clf_file:
    pickle.dump(model, clf_file)


# For test data 
y_predicted = model.predict(X_test_tfidf)
# y_predicted = clf.predict(X_test_tfidf)
# predict_prob = model.predict_proba(X_test_tfidf)

print("the following tweets are predicted to be relevant:")
print('Predicted' + '  ' + 'Test' + '  ' + 'Tweet')
print('-----------------------------------------')

est1 = open('NEALTA_train_1eST.txt', 'w', encoding="utf8")
pt2 = open('NEALTA_train_2PT.txt', 'w', encoding="utf8")
enf3 = open('NEALTA_train_3Enf.txt', 'w', encoding="utf8")
enf4 = open('NEALTA_train_4Enf.txt', 'w', encoding="utf8")
complain5 = open('NEALTA_train_5Complain.txt', 'w', encoding="utf8")
clean6 = open('NEALTA_train_6Clean.txt', 'w', encoding="utf8")
flood7 = open('NEALTA_train_7Flood.txt', 'w', encoding="utf8")
noise8 = open('NEALTA_train_8noise.txt', 'w', encoding="utf8")
traffic9 = open('NEALTA_train_9Traffic.txt', 'w', encoding="utf8")
pubsafety10 = open('NEALTA_train_10PubSafety.txt', 'w', encoding="utf8")
env11 = open('NEALTA_train_11Env.txt', 'w', encoding="utf8")
infrastructure12 = open('NEALTA_train_12Infrastructure.txt', 'w', encoding="utf8")
est13 = open('NEALTA_train_13eST.txt', 'w', encoding="utf8")
enf14 = open('NEALTA_train_14Enf.txt', 'w', encoding="utf8")
pollhaze15 = open('NEALTA_train_15PollHaze.txt', 'w', encoding="utf8")
poll16 = open('NEALTA_train_16Poll.txt', 'w', encoding="utf8")
food17 = open('NEALTA_train_17Food.txt', 'w', encoding="utf8")
pestc18 = open('NEALTA_train_18PestC.txt', 'w', encoding="utf8")
animal19 = open('NEALTA_train_19Animal.txt', 'w', encoding="utf8")
others20 = open('NEALTA_train_20Others.txt', 'w', encoding="utf8")
compliment21 = open('NEALTA_train_21Compliment.txt', 'w', encoding="utf8")
negfile = open('NEALTA_train_neg.txt', 'w', encoding="utf8")

for i in range(len(y_predicted)):
    if y_predicted[i] == '1est':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        est1.write(raw_test[i] + '\n')
    elif y_predicted[i] == '2pt':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        pt2.write(raw_test[i] + '\n')
    elif y_predicted[i] == '3enf':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        enf3.write(raw_test[i] + '\n')
    elif y_predicted[i] == '4enf':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        enf4.write(raw_test[i] + '\n')
    elif y_predicted[i] == '5complain':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        complain5.write(raw_test[i] + '\n')
    elif y_predicted[i] == '6clean':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        clean6.write(raw_test[i] + '\n')
    elif y_predicted[i] == '7flood':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        flood7.write(raw_test[i] + '\n')
    elif y_predicted[i] == '8noise':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        noise8.write(raw_test[i] + '\n')
    elif y_predicted[i] == '9traffic':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        traffic9.write(raw_test[i] + '\n')
    elif y_predicted[i] == '10pubsafety':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        pubsafety10.write(raw_test[i] + '\n')
    elif y_predicted[i] == '11env':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        env11.write(raw_test[i] + '\n')
    elif y_predicted[i] == '12infrastructure':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        infrastructure12.write(raw_test[i] + '\n')
    elif y_predicted[i] == '13est':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        est13.write(raw_test[i] + '\n')
    elif y_predicted[i] == '14enf':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        enf14.write(raw_test[i] + '\n')
    elif y_predicted[i] == '15pollhaze':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        pollhaze15.write(raw_test[i] + '\n')
    elif y_predicted[i] == '16poll':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        poll16.write(raw_test[i] + '\n')
    elif y_predicted[i] == '17food':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        food17.write(raw_test[i] + '\n')
    elif y_predicted[i] == '18pestc':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        pestc18.write(raw_test[i] + '\n')
    elif y_predicted[i] == '19animal':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        animal19.write(raw_test[i] + '\n')
    elif y_predicted[i] == '20others':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        others20.write(raw_test[i] + '\n')
    elif y_predicted[i] == '21compliment':
        print(str(y_predicted[i]) + "\t" + str(y_test[i]) + "\t" + raw_test[i])
        compliment21.write(raw_test[i] + '\n')

# positiveOutput = open('NEALTA_binary_finalpositive.txt','w', encoding="utf8") 

# for i in range(len(y_predicted)):
# 	if y_predicted[i] == 'pos':
# 		print(str(y_predicted[i])+"\t"+str(y_test[i])+"\t"+raw_test[i])
# 		positiveOutput.write(raw_test[i]+'\n')

totalp = 0
p = 0
totalr = 0
r = 0
for i in range(len(y_predicted)):
    if y_predicted[i] != 'neg':
        # if positive, add to totalp count
        totalp += 1
        if y_predicted[i] == y_test[i]:
            # if prediction is correct, add to p count
            p += 1
    if y_test[i] != 'neg':
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
