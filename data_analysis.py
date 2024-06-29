import csv
import pandas as pd

FILE_AMAZON = 'data/Amazon.csv'
FILE_GOOGLE = 'data/GoogleProducts.csv'
FILE_LINK = 'data/Amazon_GoogleProducts_perfectMapping.csv'

# Read Amazon file and do sanity checks
print('*'*50)
print('Reading Amazon file and doing sanity checks...')
data_list = []
with open(FILE_AMAZON, 'r') as f:
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        data_list.append(line)
print(len(data_list))
print(data_list[10])

cols = data_list[0]
data_list.pop(0)

dfa = pd.DataFrame(data_list, columns=cols)
print(dfa.shape)
dfa['price'] = dfa['price'].astype(float)
print(dfa.dtypes)

print('**** Checking each column ****')
print('Number of unique values for id:', dfa['id'].nunique())
print('Number of rows with missing id:', sum(pd.isnull(dfa['id']) | dfa['id'].isin([''])))

print('\nNumber of unique values for title:', dfa['title'].nunique())
print('Number of rows with missing title:', sum(pd.isnull(dfa['title']) | dfa['title'].isin([''])))
title_size = dfa.groupby(['title']).size()
title_size.sort_values(ascending=False, inplace=True)
print(title_size[0:5])

print('\nNumber of unique values for description:', dfa['description'].nunique())
print('Number of rows with missing description:', sum(pd.isnull(dfa['description']) | dfa['description'].isin([''])))
description_size = dfa.groupby(['description']).size()
description_size.sort_values(ascending=False, inplace=True)
print(description_size[0:5])

print('\nNumber of unique values for manufacturer:', dfa['manufacturer'].nunique())
print('Number of rows with missing manufacturer:', sum(pd.isnull(dfa['manufacturer']) | dfa['manufacturer'].isin([''])))
manufacturer_size = dfa.groupby(['manufacturer']).size()
manufacturer_size.sort_values(ascending=False, inplace=True)
print(manufacturer_size[0:5])
