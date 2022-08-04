import pandas as pd

"""
Given a csv file as specified in the ch_parser.py file, provides functions for
splitting the utterances into documents for clustering, saves to a csv
"""

# Reading in daata
df = pd.read_csv('docs.csv', converters={'tokens': pd.eval})

# Splitting larger df into smaller df's, each representing one file
frames = []
for doc in df.file.unique():
    frames.append(df[df.file==doc])


def partition(frame, window, step):
    """
    Takes in a data frame with a tokenized utterance column. From there combines
    the tokens for all the utterances in the window size and moves over the
    column by one step size each time. Returns a list of all the combined utterances
    """
    partitions = []
    index = 0
    for i in range(int(len(frame) / step)):
        sec = frame.tokens[index:window+index]
        sec = [token for l in sec for token in l]
        partitions.append(sec)
        index += step
    return partitions


# Creates a nx2 dataframe where the first column has the documents and the
# second column is the name of the file where the document came from
dfs = []
for f in frames:
    filename = f.file.unique()[0]
    token_frame = pd.Series(partition(f, 5, 4), name='doc_tokens').to_frame()
    token_frame['file'] = filename
    dfs.append(token_frame)
df = pd.concat(dfs, ignore_index=True)

# Saves all documents to csv
df.to_csv('documents_expanded.csv')
