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
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, Index
import json
import datetime

from emoji import emojize
warningSignEmoji = emojize(":warning:",use_aliases=True)
import telegram

es1 = Elasticsearch('http://10.0.109.44:9200')
content  = es1.search(index = 'plr_sg_tweet_201910')

text_list = []
retweet_list = []
id_list = []
for i in range (len(i['hits']['hits'])):

    #content = content['hits']['hits'][0]['_source'][0]
    text = content['hits']['hits'][i]['_source']['text']
    isRetweeted = contentts['hits']['hits'][i]['_source']['isRetweeted']
    _id = contentts['hits']['hits'][i]['_source']['id']
    text_list.append(text)
    retweet_list.append(isRetweeted)
    id_list.append(_id)

    

