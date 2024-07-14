import csv
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

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

# 1. Load a pretrained Sentence Transformer model
print('*'*80)
print('Encoding textual columns...')
model = SentenceTransformer("all-MiniLM-L6-v2")

title_a, title_b = dfA['title'], dfB['name']
desc_a, desc_b = dfA['desc_100'], dfB['desc_100']
manu_a, manu_b = dfA['manufacturer'], dfB['manufacturer']

# 2. Calculate embeddings by calling model.encode()
embeddings_title_a = model.encode(title_a)
embeddings_title_b = model.encode(title_b)
print(embeddings_title_a.shape)
print(embeddings_title_b.shape)

embeddings_desc_a = model.encode(desc_a)
embeddings_desc_b = model.encode(desc_b)
print(embeddings_desc_a.shape)
print(embeddings_desc_b.shape)

embeddings_manu_a = model.encode(manu_a)
embeddings_manu_b = model.encode(manu_b)
print(embeddings_manu_a.shape)
print(embeddings_manu_b.shape)

# 3. Calculate the embedding similarities
sim_title = model.similarity(embeddings_title_a, embeddings_title_b)
print(sim_title.shape)

sim_desc = model.similarity(embeddings_desc_a, embeddings_desc_b)
print(sim_desc.shape)

sim_manu = model.similarity(embeddings_manu_a, embeddings_manu_b)
print(sim_manu.shape)

# Save the similarity objects
with open("results/sim_title.pickle", "wb") as file:
    pickle.dump(sim_title, file)

with open("results/sim_desc.pickle", "wb") as file:
    pickle.dump(sim_desc, file)

with open("results/sim_manu.pickle", "wb") as file:
    pickle.dump(sim_manu, file)
