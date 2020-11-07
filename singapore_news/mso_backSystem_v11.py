#!/home/jjgu2/anaconda3/bin/python 
# elasticsearch_dsl.__version__=2.2.0 (note: ES DSL ver1.x does not support datetime format)
# elasticsearch.__version__=2.4.1
# Added classification step --- 20170613
# Removed Redis caching step --- 20170614
# Added location clf as 2nd step classification -- 20170621
# Refined location clf -- 20170622
# Added 2-step multiclass clf -- 20170623
# new category & sub-category test -- v8 -- 20170711
# new location detector -- v9 -- 20170828
# Add telegram function -- 20170830
# location detector with user-profiling -- 20170904
import pdb
from datetime import datetime
from ESClient import ESClient
#from redisConn import redisConn
from preprocessing import preprocessing
#from preprocessing_forLocation_v2 import *
#from locationModel_compiled_v2 import *
#from subCatFilter import findSubCategory
from elasticsearch_dsl import Search, Q
import pandas as pd
import numpy as np
#import configV2
import json
import pickle
import re
from emoji import emojize
warningSignEmoji = emojize(":warning:",use_aliases=True)
import telegram
'''
bot = telegram.Bot(token='393427009:AAFK2Qf5qpRCbLFY2A2MPMDOGIZ5pnu_jsI') # activation token gotten from step 2
def SendMessage(msg):
    #bot.send_message(chat_id="-1001126084182", text=msg, parse_mode=telegram.ParseMode.MARKDOWN) # use @channelname to target the public group
    bot.send_message(chat_id="-1001126084182", text=msg)
    return 0
'''

logfile=open('log.txt','a')
logfile.write('\n'+str(datetime.now())+' start running.'+'\n')

'''
sNum2cat = configV2.sNum2cat
print(sNum2cat)
sNum2query = configV2.sNum2query
sNum2vectorizer = configV2.sNum2vectorizer
sNum2clf = configV2.sNum2clf
sNum2preprocessConfig = configV2.sNum2preprocessConfig
'''
#additionalFilter = configV2.needAdditionalFilter
#location_Vectorizer = pickle.load(open('./Location/vectorizer_location_binary.pk',"rb"))
#location_Clf = pickle.load(open('./Location/clf_location_binary.pk',"rb"))
# location_Vectorizer = pickle.load(open('./Location_v2/tfidf_vector_f.pk',"rb"))
# location_transform2 = pickle.load(open('./Location_v2/feature_selection_f.pk',"rb"))
# location_Clf = pickle.load(open('./Location_v2/ml_model_binary_f.pk',"rb"))
# location_f = open('./Location_v2/place_final.txt','r')
# l = location_f.read().split('\n')

#cache = redisConn(host='localhost',port=6379)
#cache.clearCache

#logfile.write(str(datetime.now())+' cache cleared.'+'\n')

es1 = ESClient('http://10.0.109.44:9200')
es2 = ESClient('http://10.0.109.55:9200')
es2_indexName = 'mso_testing_v3'

logfile.write(str(datetime.now())+' connected to es1 for searching.'+'\n')

#f = Q('range',createdAt={'gte':'now+8h-15m','lte':'now+8h'})\
f = Q('range',createdAt={'lte':'now+8h'})\
	& Q("missing", field="retweetedStatus.text") & Q('term',lang='en') \
	& ~Q("term",user__name='news') & ~Q("term",user__name='newsasia') \
	& ~Q("term",user__name='times') & ~Q("term",user__name='hourly') \
	& ~Q("term",user__screenName='LTAsg') & ~Q("term",user__screenName='LTAtrafficnews') \
	& ~Q("term",user__screenName='SBSTransit_Ltd') & ~Q("term",user__screenName='SMRT_Singapore') \
	& ~Q("term",user__screenName='SingaporePolice') \
	& ~Q("term",user__screenName='NEAsg') & ~Q("term",user__screenName='PAFrenz') \
	& ~Q("term",user__screenName='AVAsg') & ~Q("term",user__screenName='URAsg') \
	& ~Q("term",user__screenName='nparksbuzz') & ~Q("term",user__screenName='PUBsingapore') \
	& ~Q("term",user__screenName='Singapore_HDB') & ~Q("term",user__screenName='MNDSingapore') \
	& ~Q("term",user__screenName='MCCYsg') & ~Q("term",user__screenName='MFAsg') \
	& ~Q("term",user__screenName='govsingapore')

#keyNum = 0
totalCount = 0
pdb.set_trace()
for sNum in sNum2query:
	#(hitCount,jresult) = es1.getSearchResult(["plr_sg_tweet_201701","plr_sg_tweet_201702","plr_sg_tweet_201703","plr_sg_tweet_201704","plr_sg_tweet_201705","plr_sg_tweet_201706","plr_sg_tweet_201707","plr_sg_tweet_201708","plr_sg_tweet_201709"],"tweet",f,sNum2query[sNum],sNum)
	(hitCount,jresult) = es1.getSearchResult("plr_sg_tweet_201708","tweet",f,sNum2query[sNum],sNum)
	#(hitCount,jresult) = es1.getSearchResult("plr_sg_tweet_live","tweet",f,sNum2query[sNum],sNum)
	print("Results for search %r." % sNum2cat[sNum])
	print('Total %d hits found.' % hitCount)
	logfile.write(str(datetime.now())+' search for '+sNum2cat[sNum]+': total hits '+str(hitCount)+'\n')
	totalCount += hitCount

	Vectorizer = pickle.load(open(sNum2vectorizer[sNum][0],"rb"))
	Clf = pickle.load(open(sNum2clf[sNum][0],"rb"))
	prepConfig = sNum2preprocessConfig[sNum][0]

	if len(sNum2clf[sNum]) == 2:
		Vectorizer2 = pickle.load(open(sNum2vectorizer[sNum][1],"rb"))
		Clf2 = pickle.load(open(sNum2clf[sNum][1],"rb"))
		prepConfig2 = sNum2preprocessConfig[sNum][1]

	logfile.write(str(datetime.now())+' Classfication in progrocess...'+'\n')
	
	lastId = es2.getTotalHit(es2_indexName,'tweet')
	count = 0
	posCount = 0

	for jdata in json.loads(jresult):

		text = jdata['text']
		if text[0:7] == "I'm at ":
			continue
		if text[0:3] == "RT ":
			continue
		user = jdata['user']
		cat = jdata['category']
		retweetCount = jdata['retweetCount']

		(rawText, processedCorpus) = preprocessing(text,user,prepConfig['wURL'],prepConfig['wHashtag'],prepConfig['wAtUser'],prepConfig['wUsername'])
		X_test = processedCorpus

		X_test_tfidf = Vectorizer.transform(X_test).toarray()
		y_predicted = Clf.predict(X_test_tfidf)
		predict_prob = Clf.predict_proba(X_test_tfidf)
		score = round(predict_prob[0][1],2)

		upload = []
		if y_predicted == 'pos':
			posCount += 1

			if len(sNum2clf[sNum]) > 1:
				(rawText, processedCorpus) = preprocessing(text,user,prepConfig2['wURL'],prepConfig2['wHashtag'],prepConfig2['wAtUser'],prepConfig2['wUsername'])
				X_test = processedCorpus
				X_test_tfidf = Vectorizer2.transform(X_test).toarray()
				y_predicted = Clf2.predict(X_test_tfidf)
				cat = str(y_predicted[0])
				#print(y_predicted)

			# Change datetime format to be ES recognizable
			fmt = '%b %d, %Y %I:%M:%S %p'
			dt = datetime.strptime(jdata['timeStamp'],fmt)
			# Addtional filter for sub category
			subCat = findSubCategory(sNum,text,cat)
			
			##### New Location Detector -- 20170904 #####
			if sNum in [1,2,3,4,5,6]:
				address = location_all(text,user)
			else:
				address = location_detector(text)
			print(address)	
			
			##### URL Detector -- 2017829 #####
			urlPattern = r'https:\/\/t\.co\/\S+'
			if re.search(urlPattern,text) != None:
				withURL = 1
			else:
				withURL = 0

			telegramMsg = user+' posted from '+jdata['platform']+' at '+ str(dt)+':\n'+text
			if address != None or withURL == 1:
				telegramMsg = warningSignEmoji+'ATTENTION'+warningSignEmoji+'\n'+telegramMsg
			#print(telegramMsg)
			#SendMessage(telegramMsg)

			upload = {
			'timeStamp':datetime.strftime(dt,'%Y-%m-%dT%H:%M:%S'),	
			'user':user,
			'platform':jdata['platform'],
			'category':subCat,	
			'agency':jdata['agency'],
			'address':address,
			'withURL':withURL,
			'retweetCount':retweetCount,
			'score':score,
			'text':rawText
			}
			jUpload = json.dumps(upload)
			if sNum in [1,2,3,4,5,6]:
				#SendMessage(telegramMsg)
				try:
					es2.indexData(es2_indexName,'tweet',count+lastId+1,jUpload)
				except:
					print('1 entry has been missed written to es server')
			else:
				if address != None:
					#SendMessage(telegramMsg)
					try:
						es2.indexData(es2_indexName,'tweet',count+lastId+1,jUpload)
					except:
						print('1 entry has been missed written to es server')
					
			logfile.write(str(datetime.now())+' '+str(posCount)+' records have been written into es2 server.'+'\n')

		count += 1

	print('--- %d positive tweets found.' % posCount)






