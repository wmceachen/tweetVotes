import pandas as pd
import re
import swifter
import matplotlib.pyplot as plt
# from twitterscraper import query_tweets_from_user
import os


def get_tweets(row):
    col = ''
    for acct in ['Campaign Twitter', 'Personal Twitter', 'Official Twitter']:
        if not pd.isna(row[acct]):
            col = acct
            break
    account = re.search("(?<=twitter\.com/).*", row[col]).group()
    os.system(f"twitterscraper {account} --user -o tweets/{account}.json")
    return account

# try:
#     tweets_df = pd.read_csv('tweets.csv')
# except:
df = pd.read_csv('accounts.csv')
df['Account Scraped'] = df.swifter.apply(get_tweets, axis=1)
tweet_dfs = []
for account in os.listdir('./tweets'):
    account_name = account.partition('.json')[0]
    account_tweets = pd.read_json('./tweets/'+account)
    account_tweets['name'] = account_name
    tweet_dfs.append(account_tweets)
all_tweets_df = pd.merge(df, pd.concat(tweet_dfs), how='right', left_on='Account Scraped', right_on='name')
all_tweets_df.to_csv('tweets/all_tweets.csv', index=False)
