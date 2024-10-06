import csv
import recordlinkage
import pandas as pd

# Files without nlp preprocessing
# FILE_AMAZON = 'data/Amazon_preprocessed.csv'
# FILE_GOOGLE = 'data/GoogleProducts_preprocessed.csv'
# FILE_LINK = 'data/Amazon_GoogleProducts_perfectMapping_preprocessed.csv'

# Files with nlp preprocessing
FILE_AMAZON = 'data/Amazon_lower_stopwords_lemmatized.csv'
FILE_GOOGLE = 'data/GoogleProducts_lower_stopwords_lemmatized.csv'
FILE_LINK = 'data/Amazon_GoogleProducts_perfectMapping_preprocessed.csv'

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
dfA['desc_100'] = [desc[:100] for desc in dfA['description']]
dfA['price'] = dfA['price'].astype(float)
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
dfB['desc_100'] = [desc[:100] for desc in dfB['description']]
dfB['price'] = dfB['price'].astype(float)
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
print('Fuzzy match on title, description, manufacturer, and price...')
compare_cl = recordlinkage.Compare(n_jobs=-1)
compare_cl.string("title", "name", method='jarowinkler', threshold=0.95, label="title")
# compare_cl.string("description", "description", method='jarowinkler', threshold=0.95, label="description")
compare_cl.string("desc_100", "desc_100", method='jarowinkler', threshold=0.95, label="desc_100")
compare_cl.string("manufacturer", "manufacturer", method='jarowinkler', threshold=0.95, label="manufacturer")
compare_cl.numeric("price", "price", method='linear', label="price")

features = compare_cl.compute(candidate_links, dfA, dfB)

print('\nDistribution of records matching on all columns')
features['price_comp'] = features['price'] >= 0.95
features['sum_all'] = features['title'] + features['desc_100'] + features['manufacturer'] + features['price_comp']
print(features['sum_all'].value_counts(ascending=False))
# features['sum_title_desc_manu'] = features['title'] + features['description'] + features['manufacturer']
features['sum_title_desc_manu'] = features['title'] + features['desc_100'] + features['manufacturer']
print(features.shape)
print(features.dtypes)
print(features.head())

# Classification step
print('*'*80)
print('Classification and evaluation for fuzzy matching...')
for match_col in [1, 2, 3]:
    if match_col == 1:
        print(f'Link if one of title, description, or manufacturer matches:')
    elif match_col == 2:
        print(f'Link if two or more of title, description, or manufacturer matches:')
    elif match_col == 3:
        print(f'Link if all of title, description, and manufacturer match:')
    matches = features[features['sum_title_desc_manu'] >= match_col]
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

print('*'*80)
print('Classification and evaluation for fuzzy matching with price...')
for match_col in [1, 2, 3, 4]:
    print(f'Link if {match_col} or more of the columns match...')
    matches = features[features['sum_all'] >= match_col]
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
