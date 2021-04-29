from urllib.request import urlopen
from collections import defaultdict
from tqdm import tqdm

import os
import json
import gzip
import pickle
import pandas as pd

DATA_DIR = '/mnt/data1/bodhi/data/amazon'

### load the meta data

data = []
with gzip.open(os.path.join(DATA_DIR, 'meta_Electronics.json.gz')) as f:
        for l in f:
            data.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)
df3 = df.fillna('')
df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows

### load the QA

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

qa_df = getDF(os.path.join(DATA_DIR, 'qa_Electronics.json.gz'))

final_df = pd.merge(df5, qa_df, on='asin')
final_df = final_df.drop_duplicates(subset=['question'])
print('Overlap for {} items'.format(len(final_df)))

only_product_df = final_df.drop_duplicates(subset=['asin'])

category_dict = defaultdict(lambda: defaultdict(int))
for i, row in tqdm(only_product_df.iterrows()):
    category = row['category']
    for i, c in enumerate(category):
        category_dict[i][c] += 1

print('Number of categories: {}'.format(len(category_dict)))

#assign clusters

assigned_cluster = []
assigned_cluster_level = []
for i, row in tqdm(final_df.iterrows()):
    category = row['category']
    for i, c in enumerate(category):
        if category_dict[i][c] < 400:
            assigned_cluster.append(c)
            assigned_cluster_level.append(i)
            break
        elif i == len(category)-1:            
            assigned_cluster.append(c)
            assigned_cluster_level.append(i)


final_df['assigned_cluster'] = assigned_cluster
final_df['assigned_cluster_level'] = assigned_cluster_level


# print('B00009R97B' in final_df['asin'])
# print('B0009NZ8CI' in final_df['asin'])
# print('B00009KLAF' in final_df['asin'])

import random
product_ids = list(set(list(only_product_df['asin'])))
random.shuffle(product_ids)
 
# split
# total_prods = len(product_ids)
# train_split = product_ids[ : int(total_prods * 0.80)]
# valid_split = product_ids[int(total_prods * 0.80) : int(total_prods * 0.90)]
# test_split = product_ids[int(total_prods * 0.90) : ]

# load from pickle
with open(os.path.join(DATA_DIR, 'train_asins.pkl'), 'rb') as f:
    train_split = pickle.load(f)
with open(os.path.join(DATA_DIR, 'valid_asins.pkl'), 'rb') as f:
    valid_split = pickle.load(f)
with open(os.path.join(DATA_DIR, 'test_asins.pkl'), 'rb') as f:
    test_split = pickle.load(f)

train_df = final_df[final_df['asin'].isin(train_split)][['asin','assigned_cluster','title','description','tech1','tech2','question', 'answer']]
valid_df = final_df[final_df['asin'].isin(valid_split)][['asin','assigned_cluster','title','description','tech1','tech2','question', 'answer']]
test_df = final_df[final_df['asin'].isin(test_split)][['asin','assigned_cluster','title','description','tech1','tech2','question', 'answer']]

print(train_df['tech1'].head())
print(train_df.head())

train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
valid_df.to_csv(os.path.join(DATA_DIR, 'valid.csv'), index=False)
test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)

# with open(os.path.join(DATA_DIR, 'train_asins.pkl'), 'wb') as f:
#     pickle.dump(train_split, f)
# with open(os.path.join(DATA_DIR, 'valid_asins.pkl'), 'wb') as f:
#     pickle.dump(valid_split, f)
# with open(os.path.join(DATA_DIR, 'test_asins.pkl'), 'wb') as f:
#     pickle.dump(test_split, f)