# -*- encoding: utf-8 -*-
# getting category_list and start-date and end-date ready
#
def ES_Query_init():
    from datetime import datetime
    from elasticsearch import Elasticsearch
    es = Elasticsearch(['http://10.0.109.55:9200'])

    query_category = es.search(
        index='mso_testing',
        body={
            "size": 0,
            "aggregations": {
                "platform_count": {
                    "terms": {"field": "category",
                              "size": 0 # add size para to unlimit return size
                              },

                }

            }
        }
    )

    query_date_max = es.search(
        index='mso_testing',
        body={
            "size": 0,
            "aggs": {
                "date_max": {
                    "max": {"field": "timeStamp"}
                }

            }
        }
    )

    query_date_min = es.search(
        index='mso_testing',
        body={
            "size": 0,
            "aggs": {
                "date_min": {
                    "min": {"field": "timeStamp"}
                }

            }
        }
    )

    # from pprint import pprint
    # content = query_category['aggregations']
    # pprint(query_category)
    # pprint(query_date_max)
    # pprint(query_date_min)

    category_list = []
    for item in query_category['aggregations']['platform_count']['buckets']:
        category_list.append(item['key'])

    date_max = datetime.strptime(query_date_max['aggregations']['date_max']['value_as_string'][:10], '%Y-%m-%d').date()
    date_min = datetime.strptime(query_date_min['aggregations']['date_min']['value_as_string'][:10], '%Y-%m-%d').date()

    print("Done get category lists and dates.")
    return category_list, date_min, date_max
