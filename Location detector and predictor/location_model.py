"""
this script is used to find location from both text and user level.
how to use:

from model_compiled import location_all
loc = location_all(textAsString, screenName)

if location is found in text, loc is purly the location string (e.g. Jurong East),
if found on user level (i.e. user profiling), loc has a prefix of "predicted" (e.g. predicted - Jurong East)
if found in neither places, returns None
"""


def location_detector(s):

    loc = None

    RULE_LIST = [
        'blk [0-9]+(|[a-z])',
        'block [0-9]+(|[a-z])',
        'blk[0-9]+(|[a-z])',
        'postal code [0-9]+',
        'bus stop [0-9]+',
    ]

    KEYWORD_LIST = [
        'Road', 'Rd',
        'Street','St',
        'Avenue', 'Ave',
        'Centre', 'Center',
        'Cafe', 'Coffee', 'Coffeshop',
        'Condo', 'Condominium',
        'Cinema',
        'Clinic',
        'Beach',
        'Club',
        'Express',
        'Hotel',
        'Hospital',
        'Plaza',
        'School',

    ]

    f = open("place_final.txt", 'r')
    l = f.read().split("\n")
    f.close()
    l += KEYWORD_LIST

    """1. Match Capital Letters"""
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()
    for num in range(len(l)):
        address = ""
        if l[num] in s:
            # print("method 1")
            """expansion on neighbors"""
            t_token = tknzr.tokenize(s)

            # pointer to the current word
            if " " in l[num]:
                try:
                    p = t_token.index(l[num].split(" ")[0])  # in case more than 2 keyword. e.g. "pasir ris"
                except ValueError:
                    break
            else:
                try:
                    p = t_token.index(l[num])
                except ValueError: # cannot find e.g. AMK's
                    break
            address += t_token[p]

            p1 = p - 1  # the previous word capitalized or number
            while p1 > 0 and (t_token[p1][0].isupper() or t_token[p1].isnumeric()):
                address = t_token[p1] + " " + address
                p1 -= 1

            p1 = p + 1  # the next word capitalized or number
            while p1 < len(t_token) - 1 and (t_token[p1][0].isupper() or t_token[p1].isnumeric()):
                address = address + " " + t_token[p1]
                p1 += 1

            return address

    """2. Regex Match"""
    import re
    for _ in RULE_LIST:
        if re.search(_, s, flags=re.IGNORECASE) != None:
            # print("method 2")
            (a, b) = re.search(_, s, flags=re.IGNORECASE).span()
            return s[a:b]

    """3. Match Complete List"""
    import csv
    with open('place-list-all.csv', 'r') as f:
        reader = csv.reader(f)
        l_all = list(reader)[0]
    for _ in l_all:
        if re.search(_, s, flags=re.IGNORECASE) != None:
            # print("method 3")
            (a, b) = re.search(_,s, flags=re.IGNORECASE).span()
            return s[a:b]

    return loc


def Sample_Text(name, size, year): # extract text for a particular user using screenName
    from elasticsearch import Elasticsearch
    if year == 2017:
        index_ = 'plr_sg_tweet_live'
    if year == 2016:
        index_ = 'plr_sg_tweet_2016'

    es = Elasticsearch(['http://kappa1.larc.smu.edu.sg:9200'], timeout=100)
    q = es.search(
        index=index_,
        body={
            "size": int(size),
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user.screenName": name}},
                    ]
                }
            },
            "filter": {
                "bool": {
                    "must": [
                        {"term": {"tweet.lang": "en"}},
                        {"missing": {"field": "tweet.retweetedStatus"}},
                    ]
                }
            }
        })

    text_list = []

    for temp in q['hits']['hits']:
        if temp["_source"]['text'].startswith("RT"):
            continue
        else:
            text_list.append(temp["_source"]['text'])

    if len(text_list) < 5:
        return None
    else:
        return text_list


def location_predictor(screenName):
	 from collections import Counter
    from fuzzywuzzy import process
    import csv

    p = Sample_Text(screenName, 1000, 2017)
    # if 2017 data not enough, scan through 2016's data
    if p == None:
        # print("2016")
        p = Sample_Text(screenName, 1000, 2016)
    elif len(p) < 10:
        # print("2016-2")
        p1 = Sample_Text(screenName, 1000, 2016)
        if p1 != None:
            p += p1

    if p == None:
        return None
    else:
        # remove empty string
        while "" in p:
            p.remove("")

        loc_pred = []
        for _ in p:
            tp = location_detector(_)
            if tp != None:
                loc_pred.append(tp)
        # print(loc_pred)

        with open('place-list-all.csv', 'r') as f:
            reader = csv.reader(f)
            l_all = list(reader)[0]

        place_std = []
        for _ in loc_pred:
            tp = process.extractOne(_, l_all)
            if tp[1] >= 50:  # Similarity check
                place_std.append(tp[0])

        if len(place_std) >= 5 and Counter(place_std).most_common(1)[0][1] >= 3:
            return "predicted - " + Counter(place_std).most_common(1)[0][0]
        else:
            return None


def location_all(text, screenName):
    loc = location_detector(text)
    if loc != None:
        return loc
    else:
        return location_predictor(screenName)