import pandas as pd
from os import listdir
from nltk.tokenize import TweetTokenizer
df = pd.concat([pd.read_csv('sent/Subtask_A/'+fn, sep='\t', header=None) for fn in listdir('sent/Subtask_A') if fn not in ['sms-2013test-A.tsv', 'livejournal-2014test-A.tsv']])
df.rename(columns = {1:'sent', 2:'tweet'}, inplace=True)
df = df[['sent','tweet']]
df.dropna(inplace=True)
df1 = pd.concat([pd.read_csv('sent/Subtask_A/sms-2013test-A.tsv', sep='\t', header=None),
pd.read_csv('sent/Subtask_A/livejournal-2014test-A.tsv', sep='\t', header=None)])
df1 = df1.rename(columns = {2:'sent', 3:'tweet'})[['sent','tweet']]
df = pd.concat([df, df1])
df.replace({'neutral': 0, 'positive': 1, 'negative':-1}, inplace=True)
tknzr = TweetTokenizer()
df.tweet = df.tweet.apply(lambda tweet: tknzr.tokenize(tweet))
# df.to_csv('sent/trinary_tweets.csv', index=False)