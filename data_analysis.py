import csv
import pandas as pd

FILE_AMAZON = 'data/Amazon.csv'
FILE_GOOGLE = 'data/GoogleProducts.csv'
FILE_LINK = 'data/Amazon_GoogleProducts_perfectMapping.csv'
FILE_AMAZON_PP = 'data/Amazon_preprocessed.csv'
FILE_GOOGLE_PP = 'data/GoogleProducts_preprocessed.csv'
FILE_LINK_PP = 'data/Amazon_GoogleProducts_perfectMapping_preprocessed.csv'

# Read Amazon file and do sanity checks
print('*'*80)
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
dfa['price'] = [''.join(list(filter(lambda x: x in '.0123456789', price))) for price in dfa['price']]
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

dfa_unique = dfa.drop_duplicates(subset=['title', 'description', 'manufacturer', 'price'], inplace=False)
print('\nNumber of unique rows:', dfa_unique.shape)
print('Writing the de-duplicated file to csv:', FILE_AMAZON_PP)
dfa_unique.to_csv(FILE_AMAZON_PP, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')

# Read Google file and do sanity checks
print('*'*80)
print('Reading Google file and doing sanity checks...')
data_list = []
with open(FILE_GOOGLE, 'r') as f:
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        data_list.append(line)
print(len(data_list))
print(data_list[10])

cols = data_list[0]
data_list.pop(0)

dfg = pd.DataFrame(data_list, columns=cols)
dfg['price'] = [''.join(list(filter(lambda x: x in '.0123456789', price))) for price in dfg['price']]
dfg['price'] = dfg['price'].astype(float)
print(dfg.shape)
print(dfg.dtypes)

print('**** Checking each column ****')
print('Number of unique values for id:', dfg['id'].nunique())
print('Number of rows with missing id:', sum(pd.isnull(dfg['id']) | dfg['id'].isin([''])))

print('\nNumber of unique values for name:', dfg['name'].nunique())
print('Number of rows with missing name:', sum(pd.isnull(dfg['name']) | dfg['name'].isin([''])))
name_size = dfg.groupby(['name']).size()
name_size.sort_values(ascending=False, inplace=True)
print(name_size[0:5])

print('\nNumber of unique values for description:', dfg['description'].nunique())
print('Number of rows with missing description:', sum(pd.isnull(dfg['description']) | dfg['description'].isin([''])))
description_size = dfg.groupby(['description']).size()
description_size.sort_values(ascending=False, inplace=True)
print(description_size[0:5])

print('\nNumber of unique values for manufacturer:', dfg['manufacturer'].nunique())
print('Number of rows with missing manufacturer:', sum(pd.isnull(dfg['manufacturer']) | dfg['manufacturer'].isin([''])))
manufacturer_size = dfg.groupby(['manufacturer']).size()
manufacturer_size.sort_values(ascending=False, inplace=True)
print(manufacturer_size[0:5])

dfg_unique = dfg.drop_duplicates(subset=['name', 'description', 'manufacturer', 'price'], inplace=False)
print('\nNumber of unique rows:', dfg_unique.shape)
print('Writing the de-duplicated file to csv:', FILE_GOOGLE_PP)
dfg_unique.to_csv(FILE_GOOGLE_PP, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')

# Read the mapping file and do sanity checks
print('*'*80)
print('Reading mapping file and doing sanity checks...')
data_list = []
with open(FILE_LINK, 'r') as f:
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        data_list.append(line)
print(len(data_list))
print(data_list[10])

cols = data_list[0]
data_list.pop(0)

dfl = pd.DataFrame(data_list, columns=cols)
print(dfl.shape)
print(dfl.dtypes)

print('**** Checking each column ****')
print('Number of unique values for idAmazon:', dfl['idAmazon'].nunique())
print('Number of rows with missing idAmazon:', sum(pd.isnull(dfl['idAmazon']) | dfl['idAmazon'].isin([''])))
idAmazon_size = dfl.groupby(['idAmazon']).size()
idAmazon_size.sort_values(ascending=False, inplace=True)
print(idAmazon_size[0:5])

print('\nNumber of unique values for idGoogleBase:', dfl['idGoogleBase'].nunique())
print('\nNumber of rows with missing idGoogleBase:', sum(pd.isnull(dfl['idGoogleBase']) | dfl['idGoogleBase'].isin([''])))
idGoogleBase_size = dfl.groupby(['idGoogleBase']).size()
idGoogleBase_size.sort_values(ascending=False, inplace=True)
print(idGoogleBase_size[0:5])

dfl_unique = dfl.drop_duplicates(subset=['idAmazon', 'idGoogleBase'], inplace=False)
dfl_unique = dfl_unique.loc[dfl_unique['idAmazon'].isin(dfa_unique['id']), :]
dfl_unique = dfl_unique.loc[dfl_unique['idGoogleBase'].isin(dfg_unique['id']), :]
print('\nNumber of unique rows:', dfl_unique.shape)
print('Writing the de-duplicated file to csv:', FILE_LINK_PP)
dfl_unique.to_csv(FILE_LINK_PP, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
