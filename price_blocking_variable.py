import csv
import pandas as pd
from collections import Counter

FILE_AMAZON = 'data/Amazon_preprocessed.csv'
FILE_GOOGLE = 'data/GoogleProducts_preprocessed.csv'
FILE_AMAZON_W_PRICE = 'data/Amazon_preprocessed_w_price_block.csv'
FILE_GOOGLE_W_PRICE = 'data/GoogleProducts_preprocessed_w_price_block.csv'


def read_file(file_path):
    print('*' * 80)
    print(f'Reading {file_path}')
    data_list = []
    with open(file_path, 'r') as f:
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            data_list.append(line)
    print(len(data_list))

    cols = data_list[0]
    data_list.pop(0)

    df = pd.DataFrame(data_list, columns=cols)
    return df


def gen_price_block(price):
    if price < 100:
        return '<100'
    elif price < 200:
        return '100-200'
    elif price < 300:
        return '200-300'
    elif price < 400:
        return '300-400'
    elif price < 500:
        return '400-500'
    else:
        return '>=500'


# Read Amazon file and store in a pandas df
df_amazon = read_file(FILE_AMAZON)
print(df_amazon.shape)
df_amazon['price'] = df_amazon['price'].astype(float)
print(df_amazon['price'].describe())
print('75%tile price: ', df_amazon['price'].quantile(0.75))
print('90%tile price:', df_amazon['price'].quantile(0.90))

df_google = read_file(FILE_GOOGLE)
print(df_google.shape)
df_google['price'] = df_google['price'].astype(float)
print(df_google['price'].describe())
print('75%tile price: ', df_google['price'].quantile(0.75))
print('90%tile price: ', df_google['price'].quantile(0.90))

# Create a blocking variable
print('*'*80)
print('Generating the blocking variable for Amazon...')
df_amazon['price_block'] = [gen_price_block(price) for price in df_amazon['price']]
print(Counter(df_amazon['price_block']))
print(df_amazon.shape)
print(df_amazon.dtypes)

print('\nGenerating the blocking variable for Google...')
df_google['price_block'] = [gen_price_block(price) for price in df_google['price']]
print(Counter(df_google['price_block']))
print(df_google.shape)
print(df_google.dtypes)

print('\nWriting the modified file to csv:', FILE_AMAZON_W_PRICE)
df_amazon.to_csv(FILE_AMAZON_W_PRICE, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')

print('Writing the modified file to csv:', FILE_GOOGLE_W_PRICE)
df_google.to_csv(FILE_GOOGLE_W_PRICE, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
