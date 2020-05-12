import pandas as pd
import re
import swifter
import matplotlib.pyplot as plt
from twitterscraper import query_tweets_from_user
df = pd.read_csv('accounts.csv')


def get_tweets(row):
    col = ''
    for acct in ['Campaign Twitter', 'Personal Twitter', 'Official Twitter']:
        if not pd.isna(row[acct]):
            col = acct
            break
    # print(row[col])
    account = re.search("(?<=twitter\.com/).*", row[col]).group()
    # print(account)
    row['tweets'] = query_tweets_from_user(account, limit=10000)
    return row

try:
    tweets_df = pd.read_csv('tweets.csv')
except:
    new_df = df.swifter.apply(get_tweets, axis=1)
    new_df.to_csv('tweets.csv', index=False)
print(tweets_df.tweets)