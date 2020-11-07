# ElasticSearch
def keyword_search(key):
    from elasticsearch import Elasticsearch
    es = Elasticsearch(['http://10.0.109.55:9200'])
    query = es.search(
        index='mso_dev1',
        body={
            "size": 3,
            "query": {
                "bool": {
                    "must": [{"wildcard": {"text": key}}],
                    "must_not": [],
                    "should": [{"exists": {"field": "address"}},
                               {"term": {"platform": "Twitter"}}
                               ],
                    "minimum_should_match": 1
                }
            },
            "sort": [
                # {"score":{"order": "desc"}},
                {"timeStamp": {"order": "desc"}},
                {"platform": {"order": "desc"}}
            ],
        }
    )
    text_list = []
    user_list = []
    timeStamp_list = []
    for _ in query['hits']['hits']:
        text_list.append(_['_source']['text'])
        user_list.append(_['_source']['user'])
        timeStamp_list.append(_['_source']['timeStamp'][:10] + ' ' + _['_source']['timeStamp'][11:])  # reformat

    return text_list, user_list, timeStamp_list