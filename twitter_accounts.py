import pandas as pd
import re
import swifter
import twitter
import matplotlib.pyplot as plt
from twitterscraper import query_tweets_from_user
# api = twitter.Api(consumer_key=[consumer key],
#                   consumer_secret=[consumer secret],
#                   access_token_key=[access token],
#                   access_token_secret=[access token secret])
earlier_data = pd.read_json("bp_accounts/2012_14.json")
later_data = pd.read_json("bp_accounts/2016_18.json")
df = pd.concat([earlier_data,later_data])
# print(df.columns.values)
df = df[['State', 'Body', 'Party', 'Year', 'Name', 'Campaign Twitter',
                   'Personal Twitter', 'Official Twitter']]
# print(df.shape)

df.dropna(how='all', inplace=True, subset=['Campaign Twitter',
                                                'Personal Twitter', 'Official Twitter'])
df.groupby("Year").size().plot(kind='bar', title='Account Totals by Year')
plt.savefig('election_counts.png')
# plt.bar(pd.value_counts(df['Year']).index, pd.value_counts(df['Year']))
# tweets = query_tweets_from_user("Palmer4Alabama", limit=10000)
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
new_df = df.iloc[:5].swifter.apply(get_tweets, axis=1)
print(new_df)