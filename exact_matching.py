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
dfA.set_index(keys=['id'], drop=True, inplace=True, verify_integrity=True)
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
dfB.set_index(keys=['id'], drop=True, inplace=True, verify_integrity=True)
print(dfB.shape)
print(dfB.dtypes)

# Read the mapping file and store in a pandas df
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

df_link = pd.DataFrame(data_list, columns=cols)
df_link.rename(columns={'idAmazon': 'id_1', 'idGoogleBase': 'id_2'}, inplace=True)
df_link.set_index(keys=['id_1', 'id_2'], drop=True, inplace=True, verify_integrity=True)
print(df_link.shape)
print(df_link.dtypes)

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
features['sum_col_match'] = features['title'] + features['description'] + features['manufacturer'] + features['price']
print(features.shape)
print(features.dtypes)
print(features.head())

# Classification step
print('*'*80)
print('Classification and evaluation for off by k matching...')
for match_col in [1, 2]:
    print(f'Link if at least {match_col} matching columns:')
    matches = features[features['sum_col_match'] >= match_col]
    print(matches.shape)
    print(matches.dtypes)
    print(matches.head())

    print('Reduction ratio:', recordlinkage.reduction_ratio(matches.index, dfA, dfB))
    print('Precision:', recordlinkage.precision(links_true=df_link, links_pred=matches.index))
    print('Recall:', recordlinkage.recall(links_true=df_link, links_pred=matches.index))
    print('F1 score:', recordlinkage.fscore(links_true=df_link, links_pred=matches.index))
    print('Confusion matrix:', recordlinkage.confusion_matrix(links_true=df_link, links_pred=matches.index,
                                                              total=len(dfA)*len(dfB)))
    print('*'*40)
