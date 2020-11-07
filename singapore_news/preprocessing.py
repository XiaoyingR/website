import string
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def preprocessing(text,user,withURL=True,withHashtag=True,withAtUser=True,withUsername=True): 
	remove = string.punctuation
	pattern = r"[{}]".format(remove) 
	
	rawText =text
	processedCorpus = []

	line = text
	if withURL == True:
		line = re.sub("https:\/\/t\.co\/\S+","wURL",line)
	else:
		pass
	if withHashtag == True:
		line = re.sub("#","wHashtag ",line)
	else:
		pass
	if withAtUser == True:
		line = re.sub("@","wAtUser ",line)
	else:
		pass
	
	line = re.sub(pattern,"",line)
	doc1 = line.split()
	doc2 = [w for w in doc1 if re.search('^[a-zA-Z]+$', w)]
	doc3 = [w.lower() for w in doc2]
	doc4 = [w for w in doc3 if len(w)>=2]
	doc5 = [stemmer.stem(w) for w in doc4]
	if withUsername == True:
		newline = " ".join(doc5) + " " + user
	else:
		newline = " ".join(doc5)
	processedCorpus.append(newline)
	
	return (rawText, processedCorpus)

