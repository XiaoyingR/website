# elasticsearch_dsl.__version__ = 2.0.0 (major change: no more F)

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, Index
import json
import datetime
'''
import configV2
sNum2cat=configV2.sNum2cat
sNum2agency=configV2.sNum2agency
sNum2query=configV2.sNum2query
'''
class ESClient:

	def __init__(self,host):
		self.client = Elasticsearch(host)

    # indexname: 表名
	def getSearchResult(self,indexName,typeName,f,q,sNum):
		s=Search(using=self.client,index=indexName,doc_type=typeName)\
			.query('bool',filter=[f])\
			.query(q)
		s = s[0:20]
		response = s.execute()
		hitCount = response.hits.total
		print(indexName)

		result = []
		num = 0
		for h in s.scan():
			if num < hitCount:
					hitdata = {
					'text':" ".join(h.text.split("\n")),
					'user':h.user.screenName,
					'timeStamp':h.createdAt,
					'retweetCount':h.retweetCount,
					'agency':sNum2agency[sNum],
					'category':sNum2cat[sNum],
					'platform':'twitter'
					}

					result.append(hitdata)
					num += 1

		jresult = json.dumps(result)
		return(hitCount,jresult)

	def matchAllSearch(self,indexName,typeName,f):
		s=Search(using=self.client,index=indexName,doc_type=typeName)\
			.query('bool',filter=[f])\
			.query(Q('match_all'))
		s = s[0:20]
		response = s.execute()
		hitCount = response.hits.total

		cat2tweet = {}
		for category in cat2tweet:
			cat2tweet[category] = []

		num = 0
		for h in s.scan():
			if num < hitCount:
				if h.category not in cat2tweet:
					cat2tweet[h.category] = []
					cat2tweet[h.category].append(h.timeStamp+'\t'+h.text)
				else:
					cat2tweet[h.category].append(h.timeStamp+'\t'+h.text)
				num+=1

		cat2count = {}
		for category in cat2tweet:
			cat2count[category] = len(cat2tweet[category])

		return(cat2count,cat2tweet)

	def createIndex(self,newIndexName):
		newIndex = Index(newIndexName,using=self.client)
		newIndex.settings(number_of_shards=1,number_of_replicas=0,mappings=None)
		#newIndex.delete(ignore=404)
		newIndex.create()

	def getTotalHit(self,indexName,typeName):
		s=Search(using=self.client,index=indexName,doc_type=typeName).query(Q('match_all'))
		response = s.execute()
		hitCount = response.hits.total
		return hitCount

	def indexData(self,indexName,typeName,id,jdata):
		self.client.index(index=indexName,doc_type=typeName,id=id,body=jdata)
