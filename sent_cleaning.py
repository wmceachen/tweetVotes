import pandas as pd
from os import listdir
from nltk.tokenize import TweetTokenizer
import swifter
from utils import clean_text
import contractions as cont
import pkg_resources
from symspellpy import SymSpell, Verbosity
import multiprocessing as mp
import numpy as np
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def sentence_spell_check(sentence):
    return sym_spell.word_segmentation(sentence).corrected_string


def parallelize_series(s, func, n_cores=mp.cpu_count()):
    pool = mp.Pool(n_cores)
    res = pool.map(func, s)
    pool.close()
    pool.join()
    return pd.Series(res)


df = pd.concat([pd.read_csv('sent/Subtask_A/'+fn, sep='\t', header=None) for fn in listdir(
    'sent/Subtask_A') if fn not in ['sms-2013test-A.tsv', 'livejournal-2014test-A.tsv']])
df.rename(columns={1: 'sent', 2: 'tweet'}, inplace=True)
df = df[['sent', 'tweet']]
df.dropna(inplace=True)
df1 = pd.concat([pd.read_csv('sent/Subtask_A/sms-2013test-A.tsv', sep='\t', header=None),
                 pd.read_csv('sent/Subtask_A/livejournal-2014test-A.tsv', sep='\t', header=None)])
df1 = df1.rename(columns={2: 'sent', 3: 'tweet'})[['sent', 'tweet']]
df = pd.concat([df, df1]).reset_index(drop=True)
df.replace({'neutral': 0, 'negative': -1, 'positive': 1}, inplace=True)
tknzr = TweetTokenizer()


def cleaning(tweet):
    return sentence_spell_check(cont.fix(clean_text(tweet)))


df.tweet = parallelize_series(df.tweet, cleaning)
df = df[(df.tweet.str.split().apply(len) > 4) &
        (df.tweet.str.split().apply(len) < 50)]
df.to_csv('sent/trinary_tweets.csv', index=False)
print(df)
