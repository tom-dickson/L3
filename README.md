# L3
A collection of files from my work with L#
## How To Use These Files
General work flow: train and save Word2Vec model with childes_full_data.py, Gather and prepare data for clustering with ch_parser.py/document_subsets.py, bring it all together for clustering and analysis with corpus_analysis.py.
### childes_full_data.py
This is where data from CHILDES Eng-NA (the all_uncleaned.json file) is read in, prepared, and fed to the Word2Vec model. The file saves both the model and the word vectors after training.
### ch_parser.py
Provides code to parse locally downloaded CHILDES .cha files into usable tabular form. Reads files line by line into a pandas dataframe, keeping track of the speaker, language, file, corpus, and pid, then saves to a csv file. 
### document_subsets.py
This file takes a dataframe as outputted by ch_parser file and generates a new dataframe where each row is a "document" ready to be clustered, along with the name of the file the document comes from. Saves to a csv.
### corpus_analysis.py
This is where the trained model comes together with the collected data for clustering. Provides functions for vectorizing doucments, clustering, and analyzing the clusters. Bottom portion of the file provides functions for clustering and analyzing a single transcription
