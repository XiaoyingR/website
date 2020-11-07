## Classifier Model Training

* **Multi**: the classifier is multi-class using positive samples across all categories
* **binary**: the classifier is binary using only positive and negative samples from one particular category

Training Process:
preprocessing (remove noise, e.g. @. #. url) -> n-gram vectorizer -> SVM