import csv
import recordlinkage
import pandas as pd

FILE_AMAZON = 'data/Amazon.csv'
FILE_GOOGLE = 'data/GoogleProducts.csv'
FILE_LINK = 'data/Amazon_GoogleProducts_perfectMapping.csv'

# Read Amazon file and store in a pandas df
print('*'*80)
print('Reading Amazon file and storing as a pandas dataframe...')
data_list = []
with open(FILE_AMAZON, 'r') as f:
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        data_list.append(line)
print(len(data_list))

cols = data_list[0]
data_list.pop(0)

dfA = pd.DataFrame(data_list, columns=cols)
print(dfA.shape)
print(dfA.dtypes)

# Read Google file and store in a pandas df
print('*'*80)
print('Reading Google file and storing as a pandas dataframe...')
data_list = []
with open(FILE_GOOGLE, 'r') as f:
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        data_list.append(line)
print(len(data_list))

cols = data_list[0]
data_list.pop(0)

dfB = pd.DataFrame(data_list, columns=cols)
print(dfB.shape)
print(dfB.dtypes)

# Indexing step
print('*'*80)
print('Indexing and running record linkage...')
indexer = recordlinkage.Index()
indexer.full()
candidate_links = indexer.index(dfA, dfB)
print("Total number of record pairs:", len(candidate_links))

# Comparison step
print('Exact match on title, description, manufacturer, and price...')
compare_cl = recordlinkage.Compare()
compare_cl.exact("title", "name", label="title")
compare_cl.exact("description", "description", label="description")
compare_cl.exact("manufacturer", "manufacturer", label="manufacturer")
compare_cl.exact("price", "price", label="price")

features = compare_cl.compute(candidate_links, dfA, dfB)

print('\nDistribution of number of columns matching exactly')
print(features.sum(axis=1).value_counts().sort_index(ascending=False))

# Classification step
matches = features[features.sum(axis=1) == 2]
print(matches.shape)
print(matches.dtypes)
print(matches.head())