#!/usr/bin/env python
# coding: utf-8

# In[212]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[233]:


states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
          'Delaware', 'Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Iowa', 'Illinois',
          'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
          'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
          'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
          'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


# In[234]:


earlier_data = pd.read_json("bp_accounts/2012_14.json")
later_data = pd.read_json("bp_accounts/2016_18.json")
tweet_df = pd.concat([earlier_data, later_data])
tweet_df


# In[235]:


tweet_df = tweet_df[['State', 'Body', 'Party', 'Year', 'Name', 'Campaign Twitter',
                     'Personal Twitter', 'Official Twitter']]
tweet_df.plot.hist('Year')


# In[236]:


def party_diff(election_df: pd.DataFrame) -> float:
    """[summary]

    Arguments:
        election_df {pd.DataFrame} -- DF with just a single election

    Returns:
        float -- p-value for difference between those who ran and those with twitter accounts
    """
    party_counts = pd.value_counts(election_df['Party'])
    election_df.dropna(how='all', inplace=True, subset=['Campaign Twitter',
                                                        'Personal Twitter', 'Official Twitter'])
    account_count = pd.value_counts(election_df.Party)
    ax = party_counts.plot.bar(title='Candidate Distribution')
    account_count.plot.bar(title='Account Distribution', ax=ax)
    plt.show()
    return stats.chisquare(account_count, party_counts)


tweet_df.groupby('Year').describe()


# In[237]:


party_counts = pd.value_counts(tweet_df['Party'])
pd.value_counts(tweet_df['Party']).plot.bar()

# tweet_df.dropna(how='all', inplace=True, subset=['Campaign Twitter',
#                                                 'Personal Twitter', 'Official Twitter'])
tweet_df.Name = tweet_df.Name.apply(
    lambda name: re.sub("\(.*\)", "", name).strip())
# pd.value_counts(df['Year']).plot.bar()


# In[238]:


pd.value_counts(tweet_df['Party']).plot.bar()
stats.chisquare(pd.value_counts(tweet_df.Party), party_counts)


# In[239]:


# 2012
def get_district_2012(race_row):
    region_info = race_row['State'].split(",")
    state, district = region_info[0].split(
        ',')[0], "".join(region_info[1:]).strip()
    return state.strip(), district.strip()


house12, senate12 = pd.read_csv(
    'elections/2012/2012_house.csv'), pd.read_csv('elections/2012/2012_senate.csv')
house12['State'], house12['District'] = zip(
    *house12.apply(get_district_2012, axis=1))
senate12['District'] = 'Senate'
elections12 = pd.concat([house12, senate12])
assert all(elections12.State.apply(lambda state: state in states))
elections12['Year'] = 2012

elections12.District.unique()


# In[240]:


# 2014
def get_district_2014(race_row):
    # print(race_row)
    if 'at-large' in race_row['District'].lower():
        state_district = race_row.District.lower().partition('at-large')
        return state_district[0].split('\'')[0].strip().title(), "".join(state_district[1:]).strip().title()
    else:
        state_district = race_row.District.partition('District')
        return state_district[0].strip(), "".join(state_district[1:]).strip()


house14, senate14 = pd.read_csv(
    'elections/2014/2014_house.csv'), pd.read_csv('elections/2014/2014_senate.csv')
house14['State'], house14['District'] = zip(
    *house14.apply(get_district_2014, axis=1))
senate14['District'] = 'Senate'
elections14 = pd.concat([house14, senate14])
elections14.rename(columns={'Total Vote': 'Total Votes'}, inplace=True)
elections14.replace({"West Virginia,": 'West Virginia', 'Louisiana Runoff Election': 'Louisiana',
                     'Oklahoma Special Election': 'Oklahoma', 'South Carolina Special Election': 'South Carolina'}, inplace=True)
assert all(elections14.State.apply(lambda state: state in states))
elections14['Year'] = 2014


def remove_special(dist_str):
    if dist_str not in ['Senate', 'At-Large District'] and not re.match('^\w{8} \d{1,2}$', dist_str):
        return re.search('^\w{8} \d{1,2}', dist_str).group()
    return dist_str


elections14.District = elections14.District.apply(remove_special)
elections14


# In[241]:


# 2016 Individual
house16, senate16 = pd.read_csv(
    'elections/2016/2016_house.csv'), pd.read_csv('elections/2016/2016_senate.csv')
house16['State'], house16['District'] = zip(
    *house16.apply(get_district_2014, axis=1))
senate16.reset_index(inplace=True)
new_header = senate16.iloc[0]  # grab the first row for the header
senate16 = senate16[1:]  # take the data less the header row
senate16.columns = new_header  # set the header row as the df header
senate16['District'] = 'Senate'


# In[242]:


# 2016 Ratings NOT USING
# def get_ratings_district(dist_data):
#     if 'at-large' in dist_data.lower():
#         state_district = dist_data.lower().partition('at-large')
#         return state_district[0].split('\'')[0].strip().title(), 'At-Large District'
#     else:
#         dist_num = re.search('\d+', dist_data).group()
#         return re.split('\d+', dist_data)[0].split('\'')[0], 'District '+dist_num
# house16_ratings = pd.read_csv('elections/2016/2016_house_ratings.csv')
# senate16_ratings = pd.read_csv('elections/2016/2016_senate_ratings.csv')
# senate16_with_ratings = senate16.merge(senate16_ratings, on='State')
# house16_ratings['State'], house16_ratings['District'] = zip(*house16_ratings.District.apply(get_ratings_district))
# # house16.merge(house16_ratings, on=['State', 'District'])
# # house16_ratings


# In[243]:


# 2016 All Elections
elections16 = pd.concat([house16, senate16])
elections16.rename(columns={'Total Vote': 'Total Votes'}, inplace=True)
assert all(elections16.State.apply(lambda state: state in states))
elections16['Year'] = 2016
elections16


# In[244]:


# 2018
def get_district_2018(dist_data):
    if 'at-large' in dist_data:
        state_district = dist_data.lower().partition('at-large')
        return state_district[0].split('\'')[0].strip().title(), 'At-Large District'
    else:
        dist_num = re.search('\d+', dist_data).group()
        return re.split('\d+', dist_data)[0].split('\'')[0], 'District '+dist_num


house18, senate18 = pd.read_csv(
    'elections/2018/2018_house.csv'), pd.read_csv('elections/2018/2018_senate.csv')
house18['State'], house18['District'] = zip(
    *house18.District.apply(get_district_2018))
senate18['State'] = senate18['District'].apply(lambda x: x.split(',')[1])
senate18['District'] = 'Senate'
elections18 = pd.concat([house18, senate18])
elections18.State = elections18.State.str.strip()
assert all(elections18.State.apply(lambda state: state in states))
elections18.rename(columns={'Runner-up': 'Top Opponent',
                            'Margin of victory': 'Margin of Victory', 'Total votes': 'Total Votes'}, inplace=True)
elections18 = elections18[['District', 'Winner',
                           'Margin of Victory',	'Total Votes',	'Top Opponent',	'State']]
elections18['Year'] = 2018
elections18


# In[245]:


elections = pd.concat([eval(f'elections{year}')for year in range(12, 20, 2)])
elections['Margin of Victory'] = elections['Margin of Victory'].str.strip(
    '%').astype(float)
assert all(elections.State.apply(lambda state: state in states))


# In[246]:


def candidate_gen(dist_row):
    region_dict = dist_row[['State', 'District', 'Year']]
    winner_dict, loser_dict = region_dict.copy(), region_dict.copy()
    winner_dict['Name'], loser_dict['Name'] = dist_row['Winner'], dist_row['Top Opponent']
    margin = dist_row['Margin of Victory']
    winner_dict['Vote Share'], loser_dict['Vote Share'] = round(
        (margin+100)/2), round(100-(margin+100)/2)
    winner, loser = pd.DataFrame(winner_dict), pd.DataFrame(loser_dict)

    return winner, loser


    # winner = pd.DataFrame({State, District,Year})
district_candidates = elections.apply(candidate_gen, axis=1).tolist()


# In[247]:


candidates = []
for district in range(len(district_candidates)):
    for i in range(2):
        candidates.append(district_candidates[district][i].transpose())


# In[248]:


candidate_df = pd.concat(candidates).reset_index(drop=True).dropna()
candidate_df = candidate_df[~candidate_df.Name.str.contains("Write|nopposed")]
candidate_df.sort_values(by=['Year', 'State', 'Name'], inplace=True)
candidate_df


# In[255]:


candidates_accounts_df = pd.merge(candidate_df, tweet_df, how='outer', on=[
                                  'Year', 'State', 'Name'], indicator=True)
candidates_accounts_df = candidates_accounts_df[candidates_accounts_df._merge == 'both'].dropna(
    how='all', subset=['Campaign Twitter',	'Personal Twitter',	'Official Twitter'])

candidates_accounts_df.to_csv('accounts.csv', index=False)