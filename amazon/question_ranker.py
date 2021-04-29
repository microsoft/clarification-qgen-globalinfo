import argparse
import csv
from collections import defaultdict
from get_schema import build_schema, build_schema_from_desc, build_schema_from_str_table
import tensorflow as tf
from tensorflow import keras
import json
from keras.models import Sequential
from keras.layers import Dense
import multiprocessing as mp
# import stanfordnlp
import stanza
import yake
import pickle as p
import random
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, confusion_matrix
import sys
import numpy as np

glove_vector_dict = None
total_glove = 0
unk_glove = 0
binary_threshold = 3.0
ds_count = 40000

# build pipelines
nlp_pipeline = stanza.Pipeline('en')
kw_extractor = yake.KeywordExtractor(n=2)

class QuestionData:
    def __init__(self, hit_id):
        self.hit_id = hit_id
        self.title = None
        self.category = None
        self.question = None
        self.question_schema = None
        self.desc_schema = None
        self.metadata_schema = None
        self.missing_info_scores = []
        self.usefulness_scores = []
    
    def add_annotations(self, missing_info_score, usefulness_score):
        self.missing_info_scores.append(missing_info_score)
        self.usefulness_scores.append(usefulness_score)

    def add_question_details(self, title, category, question, question_schema, desc_schema, metadata_schema):
        self.title = title
        self.category = category
        self.question = question
        self.question_schema = question_schema
        self.desc_schema = desc_schema
        self.metadata_schema = metadata_schema

def svr_model(X_train, y_train, X_test, y_test):
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.3))
    regr.fit(X_train, y_train)
    y_test_bar = regr.predict(X_test)
    print(mean_squared_error(y_test, y_test_bar))
    return y_test_bar
    print('True\tPredicted')
    for i in range(100):
        print('%.6f\t%.6f' % (y_test[i], y_test_bar[i]))

def svc_model(X_train, y_train, X_test, y_test, ds_X_test, ds_y_test, full_amazon_feature_set):
    # clf = make_pipeline(StandardScaler(), SVC(class_weight={0: 2, 1: 1}))
    clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf.fit(X_train, y_train)
    y_test_bar = clf.predict(X_test)
    print('F1: %.3f' % f1_score(y_test, y_test_bar))
    print('Precision: %.3f' % precision_score(y_test, y_test_bar))
    print('Recall: %.3f' % recall_score(y_test, y_test_bar))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_test_bar))

    print('DS test set scores')
    # for item in ds_X_test:
    #     try:
    #         clf.predict([item])
    #     except:
    #         import pdb; pdb.set_trace()
    ds_y_test_bar = clf.predict(ds_X_test)
    print('F1: %.3f' % f1_score(ds_y_test, ds_y_test_bar))
    print('Precision: %.3f' % precision_score(ds_y_test, ds_y_test_bar))
    print('Recall: %.3f' % recall_score(ds_y_test, ds_y_test_bar))
    print('Confusion Matrix')
    print(confusion_matrix(ds_y_test, ds_y_test_bar))

    p.dump(clf, open('question_ranker_ds.svc_model.p', 'wb'))
    print('Saved classifier')

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

def feedfoward_regr_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    # fit the keras model on the dataset
    model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=100)
    # evaluate the keras model
    results = model.evaluate(np.array(X_test), np.array(y_test))
    print("test loss, test acc:", results)
    y_test_bar = model.predict(np.array(X_test))
    print(mean_squared_error(y_test, y_test_bar))
    return y_test_bar
    print('True\tPredicted')
    for i in range(100):
        print('%.6f\t%.6f' % (y_test[i], y_test_bar[i]))

def feedfoward_clf_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=200)
    # evaluate the keras model
    results = model.evaluate(np.array(X_test), np.array(y_test))
    print("test loss, test acc:", results)
    y_test_bar = model.predict(np.array(X_test))
    print('F1: %.3f' % f1_score(y_test, y_test_bar))
    print('Precision: %.3f' % precision_score(y_test, y_test_bar))
    print('Recall: %.3f' % recall_score(y_test, y_test_bar))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_test_bar))
    return y_test_bar

def get_glove_vector(word):
    global total_glove, unk_glove
    word = word.lower()
    total_glove += 1
    if word in glove_vector_dict:
        return glove_vector_dict[word]
    # print(word)
    # import pdb; pdb.set_trace()
    unk_glove += 1
    return glove_vector_dict['unk']

def load_glove_vectors(vectors_file):
    vector_dict = {}
    for line in open(vectors_file, 'r', encoding="utf-8").readlines():
        word, vector_str = line.split(' ', 1)
        vector = [float(v) for v in vector_str.split(' ')]
        vector_dict[word] = vector
    return vector_dict

def get_asin(amazon_schema, title, category):
    for asin in amazon_schema:
        if amazon_schema[asin]['title'].replace('&amp;', '&') == title:
            return asin
    print('Failed to find asin for title: %s' % (title))

def read_mturk_data(mturk_annotations_file):
    all_questions_data = defaultdict(QuestionData)
    with open(mturk_annotations_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader: 
            hit_id = row['HITId']
            if hit_id not in all_questions_data:
                title = row['Input.title']
                category = row['Input.category']
                description = row['Input.description']
                desc_schema = build_schema_from_desc(description, kw_extractor)
                metadata = row['Input.metadata']
                metadata_schema = build_schema_from_str_table(metadata)
                question = row['Input.question']
                question_schema = build_schema(question.lower(), nlp_pipeline, kw_extractor)
                question_data = QuestionData(hit_id)
                question_data.add_question_details(title, category, question, question_schema, desc_schema, metadata_schema)
                all_questions_data[hit_id] = question_data
            for i in range(1, 6):
                if row['Answer.missing_info_%d.%d' %(i, i)] == 'true':
                    missing_info_score = i
            for i in range(6):
                if row['Answer.usefulness_%d.%d' % (i, i)] == 'true':
                    usefulness_score = i
            all_questions_data[hit_id].add_annotations(missing_info_score, usefulness_score)
    p.dump(all_questions_data, open('all_questions_data.p', 'wb')) 

def main(args):
    # if args.mturk_annotations_file:
    #     read_mturk_data(args.mturk_annotations_file)
    #     return
    amazon_schema = json.load(open(args.amazon_schema_json_file))
    
    all_questions_data = p.load(open('all_questions_data.p', 'rb'))
    global glove_vector_dict
    glove_vector_dict = load_glove_vectors(args.amazon_glove_vectors)
    feature_set = []
    labels = []
    human_true_labels = []
    human_pred_labels = []
    # mturk_questions_dict = {}
    for hit_id in all_questions_data:
        category = all_questions_data[hit_id].category
        title = all_questions_data[hit_id].title
        question_schema = all_questions_data[hit_id].question_schema
        desc_schema = all_questions_data[hit_id].desc_schema
        metadata_schema = all_questions_data[hit_id].metadata_schema
        
        if question_schema == []:
            # print('Empty schema for %s' % all_questions_data[hit_id].question)
            question_schema = [all_questions_data[hit_id].question.lower()]
        real_label = np.mean(all_questions_data[hit_id].usefulness_scores)
        label = 1 if real_label >= binary_threshold else 0

        # mturk_questions_data = {'title': title, 
        #                         'category': category,
        #                         'question': all_questions_data[hit_id].question,
        #                         'question_schema': question_schema,
        #                         'description_schema': desc_schema,
        #                         'table_schema': metadata_schema,
        #                         'real_label': real_label,
        #                         'binary_label': label}
        # mturk_questions_dict[hit_id] = mturk_questions_data

        category_vec = np.mean(np.array([get_glove_vector(w) for w in category.split(' ')]), axis=0)
        title_vec = np.mean(np.array([get_glove_vector(w) for w in title.split(' ')]), axis=0)
        desc_schema_vecs = []
        for item in desc_schema:
            if isinstance(item, str):
                desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
            elif isinstance(item, tuple):
                desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                #TODO: Assign vector for relation in tuple (currently ignored)
            else:
                assert(False)
            desc_schema_vecs.append(desc_schema_vec)
        desc_schema_vec = np.mean(desc_schema_vecs, axis=0)
        if metadata_schema == []:
            metadata_schema_vec = np.zeros(len(title_vec))
        else:
            metadata_schema_vec = np.mean(np.array([get_glove_vector(w) for w in metadata_schema]), axis=0)

        for item in question_schema:
            if isinstance(item, str):
                question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
            elif isinstance(item, tuple):
                question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                #TODO: Assign vector for relation in tuple (currently ignored)
            else:
                assert(False)
            desc_schema_vec = np.mean([desc_schema_vec, metadata_schema_vec], axis=0)
            title_vec = np.mean([category_vec, title_vec], axis=0)
            # feature_set.append(np.concatenate((category_vec, title_vec, desc_schema_vec, metadata_schema_vec, question_schema_vec), axis=0))
            feature_set.append(np.concatenate((title_vec, desc_schema_vec, question_schema_vec), axis=0))
            labels.append(label)

    print('Total: %d, 1: %d, 0: %d' % (len(labels), labels.count(1), labels.count(0)))
    print('%d out of %d were unknown tokens' % (unk_glove, total_glove))
    N = len(feature_set)
    feature_set_train = feature_set[:int(N*0.8)]
    labels_train = labels[:int(N*0.8)]
    feature_set_test = feature_set[int(N*0.8):]
    labels_test = labels[int(N*0.8):]

    count = 0
    ds_feature_set = []
    ds_labels = []
    full_amazon_feature_set = []
    empty_desc_asins = []
    for asin in amazon_schema:
        category = amazon_schema[asin]['category']
        title = amazon_schema[asin]['title']
        category_vec = np.mean(np.array([get_glove_vector(w) for w in category.split(' ')]), axis=0)
        title_vec = np.mean(np.array([get_glove_vector(w) for w in title.split(' ')]), axis=0)
        title_vec = np.mean([category_vec, title_vec], axis=0)
        desc_schema = amazon_schema[asin]['description_schema']
        metadata_schema = amazon_schema[asin]['table_schema']
        if desc_schema == [] and metadata_schema == []:
            empty_desc_asins.append(asin)
        if desc_schema == []:
            desc_schema_vec = np.zeros(len(title_vec))
        else:
            desc_schema_vecs = []
            for item in desc_schema:
                if isinstance(item, str):
                    desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
                elif isinstance(item, tuple):
                    desc_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                    #TODO: Assign vector for relation in tuple (currently ignored)
                else:
                    assert(False)
                desc_schema_vecs.append(desc_schema_vec)
            desc_schema_vec = np.mean(desc_schema_vecs, axis=0)
            
        if metadata_schema == []:
            metadata_schema_vec = np.zeros(len(title_vec))
        else:
            metadata_schema_vec = np.mean(np.array([get_glove_vector(w) for w in metadata_schema]), axis=0)

        desc_schema_vec = np.mean([desc_schema_vec, metadata_schema_vec], axis=0)   

        for i in range(len(amazon_schema[asin]['questions'])):
            question_schema = amazon_schema[asin]['questions'][i]['schema']
            if question_schema == []:
                continue            
            for item in question_schema:
                if isinstance(item, str):
                    question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
                elif isinstance(item, tuple) or isinstance(item, list):
                    question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                    #TODO: Assign vector for relation in tuple (currently ignored)
                else:
                    assert(False)
                full_amazon_feature_set.append(np.concatenate((title_vec, desc_schema_vec, question_schema_vec), axis=0))
                if i == 0:
                    count += 1
                    if count <= ds_count:
                        ds_feature_set.append(np.concatenate((title_vec, desc_schema_vec, question_schema_vec), axis=0)) 
                        ds_labels.append(1)
        
        if count <= ds_count:
            random_asin = asin
            while random_asin != asin:
                random_asin = random.choice(amazon_schema.keys())
            random_question_schema = amazon_schema[random_asin]['questions'][0]['schema']
            for item in random_question_schema:
                if isinstance(item, str):
                    random_question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item.split(' ')]), axis=0)
                elif isinstance(item, tuple) or isinstance(item, list):
                    random_question_schema_vec = np.mean(np.array([get_glove_vector(w) for w in item[:-1]]), axis=0)
                    #TODO: Assign vector for relation in tuple (currently ignored)
                else:
                    assert(False)
                ds_feature_set.append(np.concatenate((title_vec, desc_schema_vec, random_question_schema_vec), axis=0)) 
                ds_labels.append(0)

    # p.dump(full_amazon_feature_set, open('amazon_feature_set.p', 'wb'))
    print('Asins with empty desc %d out of %d' % (len(empty_desc_asins), len(amazon_schema)))
    print('Ds set stats')
    print('Total: %d, 1: %d, 0: %d' % (len(ds_labels), ds_labels.count(1), ds_labels.count(0)))
    
    c = list(zip(ds_feature_set, ds_labels))
    random.shuffle(c)
    ds_feature_set, ds_labels = zip(*c)

    ds_N = len(ds_feature_set)
    ds_feature_set_train = ds_feature_set[:int(ds_N*0.8)]
    ds_labels_train = ds_labels[:int(ds_N*0.8)]
    ds_feature_set_test = ds_feature_set[int(ds_N*0.8):]
    ds_labels_test = ds_labels[int(ds_N*0.8):]

    feature_set_train = feature_set_train + list(ds_feature_set_train)
    labels_train = labels_train + list(ds_labels_train)
    # c = list(zip(feature_set_train, labels_train))
    # random.shuffle(c)
    # feature_set_train, labels_train = zip(*c)

    # svr_preds = svr_model(feature_set_train, labels_train, feature_set_test, labels_test)
    # ffn_preds = feedfoward_model(feature_set_train, labels_train, feature_set_test, labels_test)
    full_amazon_binary_labels, full_amazon_real_labels = svc_model(feature_set_train, labels_train, feature_set_test, labels_test, ds_feature_set_test, ds_labels_test, full_amazon_feature_set)
    # ffn_clf_preds = feedfoward_clf_model(feature_set_train, labels_train, feature_set_test, labels_test)

    j = 0
    for asin in amazon_schema:
        for i in range(len(amazon_schema[asin]['questions'])):
            amazon_schema[asin]['questions'][i]['binary_label'] = full_amazon_binary_labels[j]
            amazon_schema[asin]['questions'][i]['real_label'] = full_amazon_real_labels[j]
            j += 1

    with open('amazon_schema_with_labels_ds.json', 'w') as fp:
        json.dump(amazon_schema, fp, indent=4)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--mturk_annotations_file', type = str)
    argparser.add_argument('--amazon_glove_vectors', type = str)
    argparser.add_argument('--amazon_schema_json_file', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
