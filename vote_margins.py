import pandas as pd
elections = pd.concat([pd.read_csv("old_stuff/1976-2018-house2.csv", encoding='unicode_escape'),
                       pd.read_csv("old_stuff/1976-2018-senate.csv", encoding='unicode_escape')])
elections.sort_values()
# # years = range(2012,2019,2)
# margins_df = pd.read_csv("elections/2012/2012_house.csv")
# print(margins_df.head())
# print(margins_df[margins_df.State == "Montana, At-Large, District"])
# def extract_candidates(row):
#     state_and_district = list(map(lambda word: word.strip(), row['State'].split(',')))
#     state, district = state_and_district[0], state_and_district[1]
#     print(row.index)
#     opp_row = row
# print(margins_df.apply(extract_candidates, axis=1))

# elections_file = pd.ExcelFile("old_stuff/2012congresults.xls")
# elections_df = elections_file.parse("2012 US House & Senate Results")
# print(elections_df.columns.values)
# print(elections_df['PARTY'].unique())
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
          'Delaware', 'Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
          'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
          'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
          'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
          'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
