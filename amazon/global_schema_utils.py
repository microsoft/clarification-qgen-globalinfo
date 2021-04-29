import argparse
import csv
from collections import defaultdict
from get_schema import build_schema, build_schema_from_desc, build_schema_from_str_table
import json
import multiprocessing as mp
import pickle as p
import random
import sys
import numpy as np

glove_vector_dict = {}

def get_glove_vector(word):
    word = word.lower()
    if word in glove_vector_dict:
        return glove_vector_dict[word]
    return glove_vector_dict['unk']

def load_glove_vectors(vectors_file):
    vector_dict = {}
    for line in open(vectors_file, 'r', encoding="utf-8").readlines():
        word, vector_str = line.split(' ', 1)
        vector = [float(v) for v in vector_str.split(' ')]
        vector_dict[word] = vector
    return vector_dict

def create_keyword_embs(schema):
    for asin in schema:
        ct += 1
        if ct > 11:
            break
        
        # doing for only one prototype schema, do for all likewise
        schema = schema[asin]['schema']
        schema_map = []
        schema_vecs = []
        for item in schema:
            if isinstance(item, str):
                schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
                kphrase = item.lower()
            elif isinstance(item, tuple):
                schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                kphrase = item[:-1].lower()
                #TODO: Assign vector for relation in tuple (currently ignored)
            else:
                assert(False)
            schema_map.append(kphrase)
            schema_vecs.append(schema_vec)

    return schema_map, schema_vec

def create_clusters(schema_vecs, schema_map):
    '''
    Gives you a dict with keyphrase and it's cluster label
    Use these cluster labels to group similar keyphrases
    '''

    kp_clusters = {}
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(distance_threshold=0.6, n_clusters=None)

    model = model.fit(np.array(schema_vecs))

    for i, l in enumerate(list(model.labels_)):
        kp_clusters[schema_map[i]] = l

    return kp_clusters










