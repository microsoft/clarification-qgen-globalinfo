import argparse
import csv
from collections import defaultdict
from get_schema import build_schema, build_schema_from_desc, build_schema_from_str_table
import json
import multiprocessing as mp
import pickle as p
import random
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, confusion_matrix
import sys
import numpy as np

glove_vector_dict = None

def clf_predict(clf, feature_set):
    return clf.predict_proba(feature_set)

def svc_model_predict(clf, full_amazon_feature_set):
    full_amazon_pred_binary_labels = clf.predict(full_amazon_feature_set)
    print('Done predicting binary labels')
    N = len(full_amazon_feature_set)
    num_cpus = mp.cpu_count()
    print('No. of cpus %d' % num_cpus)
    batch_size = 1000

    pool = mp.Pool(num_cpus)
    pred_probs = []
    for i in range(int(N/batch_size)+1):
        pred_probs += list(pool.apply(clf_predict, args=(clf, full_amazon_feature_set[i*batch_size:(i+1)*batch_size])))

    # pred_probs = clf.predict_proba(full_amazon_feature_set)
    print('Done predicting probabilities')
    full_amazon_pred_real_labels = []
    for i in range(len(full_amazon_pred_binary_labels)):
        if full_amazon_pred_binary_labels[i] == 0:
            real_label = min(pred_probs[i])
        elif full_amazon_pred_binary_labels[i] == 1:
            real_label = max(pred_probs[i])
        else:
            assert(False)
        full_amazon_pred_real_labels.append(real_label)
    print('Done predicting real labels')
    return full_amazon_pred_binary_labels, full_amazon_pred_real_labels

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

def main(args):
    amazon_schema = json.load(open(args.amazon_schema_json_file))
    full_amazon_feature_set = p.load(open(args.amazon_feature_set_pickle_file, 'rb'))
    clf = p.load(open(args.svc_model_pickle_file, 'rb'))
    
    # global glove_vector_dict
    # glove_vector_dict = load_glove_vectors(args.amazon_glove_vectors)

    # print('Extracting features from amazon data... ')
    # full_amazon_feature_set = []
    # ct = 0
    # for asin in amazon_schema:
    #     ct += 1
    #     if ct > 11:
    #         break
    #     category = amazon_schema[asin]['category']
    #     title = amazon_schema[asin]['title']
    #     category_vec = np.mean(np.array([get_glove_vector(w) for w in category.split(' ')]), axis=0)
    #     title_vec = np.mean(np.array([get_glove_vector(w) for w in title.split(' ')]), axis=0)
    #     title_vec = np.mean([category_vec, title_vec], axis=0)
    #     desc_schema = amazon_schema[asin]['description_schema']
    #     metadata_schema = amazon_schema[asin]['table_schema']
    #     if desc_schema == []:
    #         desc_schema_vec = np.zeros(len(title_vec))
    #     else:
    #         desc_schema_vecs = []
    #         for item in desc_schema:
    #             if isinstance(item, str):
    #                 desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
    #             elif isinstance(item, tuple):
    #                 desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
    #                 #TODO: Assign vector for relation in tuple (currently ignored)
    #             else:
    #                 assert(False)
    #             desc_schema_vecs.append(desc_schema_vec)
    #         desc_schema_vec = np.mean(desc_schema_vecs, axis=0)
    #     if metadata_schema == []:
    #         metadata_schema_vec = np.zeros(len(title_vec))
    #     else:
    #         metadata_schema_vec = np.mean(np.array([get_glove_vector(w) for w in metadata_schema]), axis=0)

    #     desc_schema_vec = np.mean([desc_schema_vec, metadata_schema_vec], axis=0)    
    #     for i in range(len(amazon_schema[asin]['questions'])):
    #         question_schema = amazon_schema[asin]['questions'][i]['schema']
    #         if question_schema == []:
    #             continue            
    #         for item in question_schema:
    #             if isinstance(item, str):
    #                 question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
    #             elif isinstance(item, tuple) or isinstance(item, list):
    #                 question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
    #                 #TODO: Assign vector for relation in tuple (currently ignored)
    #             else:
    #                 assert(False)
    #             full_amazon_feature_set.append(np.concatenate((title_vec, desc_schema_vec, question_schema_vec), axis=0))
    # p.dump(full_amazon_feature_set, open('amazon_feature_set.p', 'rb'))
    # print('Done!')
    full_amazon_binary_labels, full_amazon_real_labels = svc_model_predict(clf, full_amazon_feature_set)

    j = 0
    for asin in amazon_schema:
        for i in range(len(amazon_schema[asin]['questions'])):
            if amazon_schema[asin]['questions'][i]['schema'] == []:
                amazon_schema[asin]['questions'][i]['binary_label'] = None
                amazon_schema[asin]['questions'][i]['real_label'] = None
            else:
                k = len(amazon_schema[asin]['questions'][i]['schema'])
                if full_amazon_binary_labels[j:j+k].tolist().count(1) >= full_amazon_binary_labels[j:j+k].tolist().count(0):
                    binary_label = 1 
                else:
                    binary_label = 0
                amazon_schema[asin]['questions'][i]['binary_label'] = binary_label
                amazon_schema[asin]['questions'][i]['real_label'] = sum(full_amazon_real_labels[j:j+k])/k
                j += k

    p.dump(amazon_schema, open('amazon_schema_with_labels.p', 'wb'))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--amazon_schema_json_file', type = str)
    argparser.add_argument('--svc_model_pickle_file', type = str)
    argparser.add_argument('--amazon_feature_set_pickle_file', type = str)
    # argparser.add_argument('--amazon_glove_vectors', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)