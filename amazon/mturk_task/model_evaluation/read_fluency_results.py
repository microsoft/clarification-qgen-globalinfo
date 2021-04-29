import argparse
import csv
from collections import defaultdict
import numpy as np
import sys

def main(args):
    fluent_scores = defaultdict(list)
    fluent_score_histogram = defaultdict(int)
    model_scores_by_hit = {'bart': defaultdict(list), 'bart_misinfo': defaultdict(list), 'bart_misinfo_pplm': defaultdict(list), 'ref': defaultdict(list), 'gan': defaultdict(list)}
    model_scores = {'bart': [], 'bart_misinfo': [], 'bart_misinfo_pplm': [], 'ref': [], 'gan': []}
    worker_scores = defaultdict(list)
    with open(args.mturk_batch1_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.fluent_%d.%d' % (i, i)] == 'true':
                    fluent_scores[hit_id].append(i)
                    fluent_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)
    
    with open(args.mturk_batch2_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.fluent_%d.%d' % (i, i)] == 'true':
                    fluent_scores[hit_id].append(i)
                    fluent_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)

    with open(args.mturk_batch3_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.fluent_%d.%d' % (i, i)] == 'true':
                    fluent_scores[hit_id].append(i)
                    fluent_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)

    with open(args.mturk_batch4_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.fluent_%d.%d' % (i, i)] == 'true':
                    fluent_scores[hit_id].append(i)
                    fluent_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)

    for model in model_scores_by_hit:
        for hit_id in model_scores_by_hit[model]:
            if model_scores_by_hit[model][hit_id].count(1) > model_scores_by_hit[model][hit_id].count(0):
                model_scores[model].append(1)
            else:
                model_scores[model].append(0)

    # print('\n\n')
    fluent_score_var = 0
    fluent_agreement = 0
    for hit_id in fluent_scores:
        fluent_score_var += np.var(fluent_scores[hit_id])
        for i in range(2):
            if fluent_scores[hit_id].count(i) > 1:
            # if fluent_scores[hit_id].count(i) > 2 or (fluent_scores[hit_id].count(i) + fluent_scores[hit_id].count(i-1)) > 2 or (fluent_scores[hit_id].count(i) + fluent_scores[hit_id].count(i+1)) > 2:
                fluent_agreement += 1
                break

    print('fluent score variance %.2f' % (fluent_score_var*1./len(fluent_scores)))
    print('fluent agreement: %d out of %d' % (fluent_agreement, len(fluent_scores)))

    for model in model_scores:
        print('%s: %.2f' % (model, sum(model_scores[model])*1.0/len(model_scores[model])))

    print('fluent score histogram')
    print(fluent_score_histogram)

    # for worker_id in worker_scores:
    #     print(worker_id)
    #     print(worker_scores[worker_id])
    #     print(worker_scores[worker_id].count(2))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mturk_batch1_results_file", type = str)
    argparser.add_argument("--mturk_batch2_results_file", type = str)
    argparser.add_argument("--mturk_batch3_results_file", type = str)
    argparser.add_argument("--mturk_batch4_results_file", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)