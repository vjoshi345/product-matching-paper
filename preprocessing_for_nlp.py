import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd

FILE_AMAZON = 'data/Amazon_preprocessed.csv'
FILE_GOOGLE = 'data/GoogleProducts_preprocessed.csv'
FILE_AMAZON_PP = 'data/Amazon_lower_stopwords_lemmatized.csv'
FILE_GOOGLE_PP = 'data/GoogleProducts_lower_stopwords_lemmatized.csv'


def preprocess_str(inp_str):
    wnl = nltk.WordNetLemmatizer()

    # Remove whitespace from ends and convert to lowercase
    inp_str = inp_str.strip()
    inp_str = inp_str.lower()

    # Remove stopwords
    inp_list = inp_str.split(' ')
    inp_list = [word for word in inp_list if word not in stopwords.words('english')]

    # Lemmatization
    inp_list = [wnl.lemmatize(word) for word in inp_list]

    out_str = ' '.join(inp_list)
    return out_str


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

print('\nLowercase, remove stopwords, and lemmatize the columns...')
dfA['title'] = [preprocess_str(s) for s in dfA['title']]
dfA['description'] = [preprocess_str(s) for s in dfA['description']]
dfA['manufacturer'] = [preprocess_str(s) for s in dfA['manufacturer']]
print(dfA.shape)
print(dfA.dtypes)

print('Writing the modified file to csv:', FILE_AMAZON_PP)
dfA.to_csv(FILE_AMAZON_PP, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')

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

print('\nLowercase, remove stopwords, and lemmatize the columns...')
dfB['name'] = [preprocess_str(s) for s in dfB['name']]
dfB['description'] = [preprocess_str(s) for s in dfB['description']]
dfB['manufacturer'] = [preprocess_str(s) for s in dfB['manufacturer']]
print(dfB.shape)
print(dfB.dtypes)

print('Writing the modified file to csv:', FILE_GOOGLE_PP)
dfB.to_csv(FILE_GOOGLE_PP, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
