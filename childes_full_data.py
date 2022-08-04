import json
import nltk

from nltk.corpus import stopwords
from gensim.models import Word2Vec

"""
Training word2vec model on full set of childes data
"""

# reading in file
with open("all_uncleaned.json") as file:
    data = json.loads(file.read())

# removing stopwords and tokenizing
stops = set(stopwords.words('english'))
remove_stops = lambda l: [w for w in l if not w in stops]
data = [remove_stops(l) for l in data]

# Training mmodel
model = Word2Vec(sentences=data, vector_size=200)

# Saving
model.wv.save("all_data.wordvectors")
model.save("all_data.model")
