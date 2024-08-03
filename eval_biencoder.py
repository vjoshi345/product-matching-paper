import csv
import pickle
import recordlinkage
import pandas as pd

SIM_TITLE_FILE = 'results/sim_title.pickle'
SIM_DESC_FILE = 'results/sim_desc.pickle'
SIM_MANU_FILE = 'results/sim_manu.pickle'

with open(SIM_TITLE_FILE, 'rb') as file:
    sim_title = pickle.load(file).numpy()
    sim_title = sim_title.flatten(order='C')
    print(sim_title.shape)

with open(SIM_DESC_FILE, 'rb') as file:
    sim_desc = pickle.load(file).numpy()
    sim_desc = sim_desc.flatten(order='C')
    print(sim_desc.shape)

with open(SIM_MANU_FILE, 'rb') as file:
    sim_manu = pickle.load(file).numpy()
    sim_manu = sim_manu.flatten(order='C')
    print(sim_manu.shape)

FILE_AMAZON = 'data/Amazon_preprocessed.csv'
FILE_GOOGLE = 'data/GoogleProducts_preprocessed.csv'
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

# Get the candidate links and then fill up matches for each column
print('*'*80)
print('Getting candidate links and filling up match status for each column...')
df_cross = pd.merge(dfA[['id']], dfB[['id']], how='cross', suffixes=('_1', '_2'))
df_cross.set_index(keys=['id_1', 'id_2'], drop=True, inplace=True, verify_integrity=True)
df_cross['sim_title'] = sim_title
df_cross['sim_desc'] = sim_desc
df_cross['sim_manu'] = sim_manu
for th in [0.9, 0.8, 0.7]:
    for col in ['sim_title', 'sim_desc', 'sim_manu']:
        df_cross[f'{col}_gt_{th}'] = df_cross[col] >= th
        print(sum(df_cross[f'{col}_gt_{th}']))
print(df_cross.shape)
print(df_cross.dtypes)

# Classification step
print('*'*80)
print('Classification and evaluation for fuzzy matching...')
for th in [0.9, 0.8, 0.7]:
    for col in ['sim_title', 'sim_desc', 'sim_manu']:
        match_col = f'{col}_gt_{th}'
        print(f'Matching on {match_col}')
        matches = df_cross[df_cross[match_col] == 1]
        print(matches.shape)

        print('Reduction ratio:', recordlinkage.reduction_ratio(matches.index, dfA, dfB))
        print('Precision:', recordlinkage.precision(links_true=df_link, links_pred=matches.index))
        print('Recall:', recordlinkage.recall(links_true=df_link, links_pred=matches.index))
        print('F1 score:', recordlinkage.fscore(links_true=df_link, links_pred=matches.index))
        print('Confusion matrix:', recordlinkage.confusion_matrix(links_true=df_link, links_pred=matches.index,
                                                                  total=len(dfA)*len(dfB)))
        print('*'*40)
