from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

"""
This file takes the documents as saved created in the document_subsets.py file
and provides functions to analyze/cluster them using a locally saved word2vec model.
"""

brown = pd.read_csv('documents_expanded.csv', converters={'doc_tokens': pd.eval})
brown.drop(columns=['Unnamed: 0'], inplace=True)

print(brown.shape)

def utt_sizes(frame):
    """
    Plots a histogram of the distribution of the length of each utterance in the
    data frame
    """
    sizes = []
    for l in frame.doc_tokens:
        sizes.append(len(l))
    plt.hist(sizes)
    plt.show()

print("----------\nLoading Model\n---------")
#word_vecs = api.load("fasttext-wiki-news-subwords-300")
#model = Word2Vec.load("all_data.model") # Loading Word2vec model
word_vecs = KeyedVectors.load("all_data.wordvectors") # Loading word vectors
print("----------\nModel Loaded\n---------")


def get_doc_vec(doc):
    """
    Turns an utterance into a vector using Word2Vec model
    """
    vec = np.zeros(200)
    i = 0
    for word in doc:
        try:
            vec = np.add(vec, word_vecs[word])
            i += 1
        except KeyError:
            continue
    return vec


def choose_k(arr, n):
    """
    Plots line graph for finding optimal number of clusters
    """
    ssd = []
    clusters = range(1, n)
    for k in clusters:
        km = KMeans(n_clusters=k)
        km.fit(arr)
        ssd.append(km.inertia_)
    plt.plot(clusters, ssd)
    plt.xlabel('k value')
    plt.ylabel('Sum of squared distance')
    plt.show()


# Gets a list of the vectors representing each utterance in the original dataframe
vecs =[]
for doc in brown.doc_tokens:
    vecs.append(get_doc_vec(doc))

# nx200 array  where  each row is an utterance vector
arr = np.array(vecs)

print("----------\nClustering\n---------")
# Training k-Means model
km = KMeans(n_clusters=40)
clusters = km.fit_predict(arr)

# Creating data frame where first 200 columns are the utterance vectors and there
# are 2 additional columns, the cluster and the tokenized utterance
df = pd.DataFrame(arr, columns=range(1, 201))
df['cluster'] = clusters
df['tokens'] = brown['doc_tokens']


def plot_clusters(frame, sample_size, annotate=False):
    """
    Reduces data to 2 dimensions using PCA then plots clusters, sample_size
    is an integer specifying how many data points to plot. The annotate parameter
    chooses whether or not to annotate each point with the word it represents
    """
    colors = iter(cm.rainbow(np.linspace(0, 1, len(frame['cluster'].unique()))))
    df = frame.drop(columns=['cluster', 'tokens'])
    arr = df.to_numpy()
    pca = PCA(n_components=2)
    pca.fit(arr)
    arr_2d = pca.transform(arr)
    df_2d = pd.DataFrame(arr_2d, columns = ['pc1', 'pc2'])
    df_2d['cluster'] = frame['cluster']
    sample = df_2d.sample(n=sample_size)
    for c in sample.cluster.unique():
        subset = sample[sample['cluster']==c]
        plt.scatter(subset['pc1'], subset['pc2'], color=next(colors), s=4)
        if annotate:
            for id, row in sample.iterrows():
                plt.annotate(row['tokens'], (row['pc1'], row['pc2']),
                            fontsize=3, alpha=.6)
    plt.show()


def closest_to_centroid(cluster: int, n):
    """
    Returns the n documents which are closest to the specified centroid
    """
    subset = df[df.cluster == cluster]
    center = km.cluster_centers_[cluster,:]
    distances = []
    for i in range(subset.shape[0]):
        row_vec = df.iloc[i, 0:200].to_numpy()
        dist = np.linalg.norm(row_vec - center)
        distances.append((i, dist))
    n_sorted_dists = sorted(distances, key=lambda x: x[1])
    for i in n_sorted_dists[:n]:
        indx = i[0]
        line = df.iloc[[indx]]
        print(line.tokens)


def print_cluster_lines(frame, cluster, n: int):
    """
    Prints a sample of the utterances in a given cluster
    """
    subset = frame[frame.cluster == cluster]
    print(subset.tokens.sample(n))


def n_most_common_words(frame, cluster, n: int):
    """
    Plots/prints the n most common words in a specified cluster
    """
    subset = frame[frame.cluster==cluster].tokens
    counts = Counter([word for utt in subset for word in utt])
    sorted_counts = dict(sorted(counts.items(), key=lambda x:x[1],reverse=True))
    plt.bar(range(n), list(sorted_counts.values())[:n])
    plt.xticks(range(n),list(sorted_counts.keys())[:n])
    plt.title(f'cluster {cluster}')
    plt.show()


def get_cluster_topic(frame, cluster):
    """
    Given a cluster, returns 5 words which capture the topic of the cluster
    """
    subset = frame[frame.cluster==cluster].tokens
    counts = Counter([word for utt in subset for word in utt])
    sorted_counts = dict(sorted(counts.items(), key=lambda x:x[1],reverse=True))
    highest_words = list(sorted_counts.keys())[:15]
    most_similar_lists = []
    for word in highest_words:
        try:
            most_similar_lists.append(word_vecs.most_similar(word))
        except KeyError:
            continue
    most_similar_list = [x for l in most_similar_lists for x in l]
    words = set([w for w, v in most_similar_list])
    scores = []
    for w in words:
        sims = [v for word, v in most_similar_list if word == w]
        scores.append((w, sum(sims)))
    sorted_scores = sorted(scores, key=lambda x:x[1],reverse=True)
    return sorted_scores[:6]


def analyze_clusters(list_of_clusters=df.cluster.unique()):
    for i in list_of_clusters:
        try:
            print(f"****** \n cluster {i} \n******")
            closest_to_centroid(i, 10)
            n_most_common_words(df, i, 20)
            print(get_cluster_topic(df, i))
        except ValueError:
            continue

"""
Below are functions for finding which topics arise in a single transcription
"""

def get_tokens(line):
    """
    Helper function for tokenizing/removing stopwords from an utterance
    """
    tokenizer = RegexpTokenizer(r"[\w']+")
    stops = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(line)
    return [w for w in tokens if not w in stops]


def get_transcription(path):
    """
    Parses .cha file into dataframe
    """
    with open(f'{path}', 'r') as file:
        lines = []
        for line in file.readlines():
            if line.startswith('@Languages'):
                lang = line.split('\t')[1].strip()
            elif line.startswith('@PID'):
                pid = line.split('\t')[1].strip()
            elif line.startswith('*'):
                line = re.sub('[*,:,\n]', '', line)
                split = line.split('\t')
                lines.append((split[1], split[0]))
    df = pd.DataFrame(lines, columns=['line', 'speaker'])
    df['tokens'] = df['line'].apply(get_tokens)
    df['language'] = lang
    df['pid'] = pid
    return df


def partition(frame, window, step):
    """
    Takes in a data frame with a tokenized utterance column. From there combines
    the tokens for all the utterances in the window size and moves over the
    column by one step size each time. Returns a df of all the combined utterances
    """
    partitions = []
    index = 0
    for i in range(int(len(frame) / step)):
        sec = frame.tokens[index:window+index]
        sec = [token for l in sec for token in l]
        partitions.append(sec)
        index += step
    return pd.Series(partitions, name='doc_tokens').to_frame()


def find_topics(frame):
    """
    Given a df with one doc_tokens column (a transcription broken up into pieces)
    returns a list of tuples (a, b) where a is a cluster and b is the number of
    documents from the transcription which fall into that cluster
    """
    vecs = []
    for doc in frame.doc_tokens:
        vecs.append(get_doc_vec(doc))
    arr = np.array(vecs)
    clusters = km.predict(arr).tolist()
    unique = set(clusters)
    counts = [clusters.count(i) for i in unique]
    print(list(zip(unique, counts)))
    return list(zip(unique, counts))


def get_topics(path_to_file, window, step):
    """
    General function which takes the path to a .cha file and returns the topics
    """
    df = get_transcription(path_to_file)
    docs = partition(df, window, step)
    return find_topics(docs)


doc_clusters = get_topics('/Users/tomdickson/Desktop/childes_data/bliss/normelis.cha', 5, 4)
for i in [c[0] for c in doc_clusters]:
    analyze_clusters([i])
