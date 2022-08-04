import requests
import re
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

"""
Provides functions for parsing .cha files downloaded from CHILDES into a DataFrame
and to a csv file
"""

def get_tokens(line):
    """
    Helper function for tokenizing/removing stopwords from an utterance
    """
    tokenizer = RegexpTokenizer(r"[\w']+")
    stops = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(line)
    return [w for w in tokens if not w in stops]


def collect_transcriptions(path, corpus_name: str):
    """
    Function for generating a pandas df from locally downloaded .cha files. Takes as
    input the path to the directory containing the files and parses each file
    line by line to put the transcriptions in tabular form. Returns the transcriptions
    as a dataframe where each row is an utterance labeled by who spoke it, which file
    /corpus it is in, the pid, and the language.
    """
    chi_files = os.listdir(path)
    frames = []
    for f in chi_files:
        with open(f'{path}/{f}', 'r') as file:
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
            df['file'] = f
            df['pid'] = pid
            df['corpus'] = corpus_name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df

# Collecting transcriptions from locally downloaded cha .files
df1 = collect_transcriptions('/Users/tomdickson/Desktop/childes_data/gleason_dinner', 'Gleason')
df2 = collect_transcriptions('/Users/tomdickson/Desktop/childes_data/brown_adam', 'Brown')
df3 = collect_transcriptions('/Users/tomdickson/Desktop/childes_data/brown_eve', 'Brown')
df4 = collect_transcriptions('/Users/tomdickson/Desktop/childes_data/brown_sarah', 'Brown')
df5 = collect_transcriptions('/Users/tomdickson/Desktop/childes_data/bliss', 'Bliss')

# Concatenating data frames
dfs = [df1, df2, df3, df4, df5]
frame = pd.concat(dfs, ignore_index=True)
print(frame.shape)
frame.to_csv('docs.csv')
