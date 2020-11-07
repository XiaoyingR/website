## Location Detector and Predictor
#### necessary library & external files:
  * 2 lists of singapore-based locations: one for fuzzy matching (place_final.txt) and one for exact match (place-list-all.csv)
  * nltk, Elasticsearch, fuzzywuzzy
#### process procedure
  * the model takes in a text and the user's screenName. 
  * If location is found in **text**, location string (e.g. Jurong East) is returned; if not, it jumps into the historical tweets of that particular user (i.e. **user profiling**), and returns with a prefix of "predicted" (e.g. predicted - Jurong East). If nothing is found in neither places, returns None.
#### Text-level matching (location_detector)
3 rounds of search:
  * 1st: from fuzzy matching extract those with **Capitalized Letters** (e.g. Jurong East)
  * 2nd: use **Regex** to extract certain patterns (e.g. blk 221A)
  * 3rd: use exact match to extract those without capitalized letters (e.g. jurong east)
#### User-level Profiling (location_predictor)
* only user that tweets more than 10 in the year of 2016 plus 2017 would be considered as valid
* then, contrained to a sample size maximumly 1000, use **text-level exact matching** to extract locations.
* Counter to extract the most frequently mention location.
#### How-to-Use
```python
from location_model import location_all
loc = location_all(textAsString, screenName)

# or separately
from location_model import *
loc_text = location_detector(textAsString)
loc_user = location_predictor(screenName)
```