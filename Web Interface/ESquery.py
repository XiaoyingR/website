# -*- encoding: utf-8 -*-
#
def ES_Query(category_choice, date_choice_start, date_choice_end, RANK, MAX_SIZE):


    print(category_choice, date_choice_start, date_choice_end)
    import pandas as pd
    import elasticsearch.helpers as helpers
    from elasticsearch import Elasticsearch

    if category_choice=="All":
        query = {"query": {
                   "range": {
                       "timeStamp": {
                           "gte": date_choice_start,
                           "lte": date_choice_end
                       }
                   }
        }}
    else:
        query = {"query": {
                    "and": [
                        {
                            "range": {
                                "timeStamp": {
                                     "gte": date_choice_start,
                                     "lte": date_choice_end
                                }
                            }
                        },
                        {
                            "bool": {
                                "must": [
                                    {"term": {"category": category_choice}}
                                ],
                                "must_not": [],
                                "should": []
                            }
                        }
                    ]
        }}


    # query ={"query": {
    #     "match_all": {}
    # }}
    # if QUERY_TYPE == "platform":
    #     query = {"query": {
    #         "bool": {
    #             "must": [
    #                 {"term": {"platform": 'twitter'}},
    #             ],
    #             "should": [],
    #             "minimum_should_match": 1,
    #         }
    #     }}
    # if QUERY_TYPE == 'mixed':
    #     query = {
    #         "size": 0,
    #         "aggs": {
    #             "platform_count": {
    #                 "terms": {"field": "platform"}
    #         }
    #     }
    #     }
    es = Elasticsearch(['http://10.0.109.55:9200'])
    index='mso_testing'
    num=0

    location_list = []
    timeStamp_list = []
    platform_list = []
    category_list = []
    text_list = []
    score_list = []
    agency_list = []

    json = helpers.scan(client=es, query=query, index=index, size=1000, request_timeout=None)
    for raw in json:
        if num < MAX_SIZE:
            try:
                location_list.append(raw["_source"]['address'])
            except KeyError:
                location_list.append(None)
            timeStamp_list.append(raw['_source']['timeStamp'])
            platform_list.append(raw['_source']['platform'])
            category_list.append(raw['_source']['category'])
            text_list.append(raw['_source']['text'])
            score_list.append(raw['_source']['score'])
            agency_list.append(raw['_source']['agency'])
        else:
            break
        num = num + 1


    # Create pandas frame
    data = pd.DataFrame({
        'location': location_list,
        'timeStamp': timeStamp_list,
        'platform': platform_list,
        'category': category_list,
        'text': text_list,
        'score': score_list,
        'agency': agency_list

    })
    if RANK == 'timeStamp':
        # data.sort_values(by='timeStamp', ascending=False, inplace=True)
        data.sort(['timeStamp', 'platform'], ascending=[False,False], inplace=True)
    if RANK == 'score':
        data.sort_values(by='score', ascending=False, inplace=True)

    # data.sort_values(by='location',ascending=False,inplace=True)

    print("Query done. Length of:",len(data))
    # print("sample data: \n", data.head(3))
    return data