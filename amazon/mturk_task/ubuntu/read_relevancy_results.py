import argparse
import csv
from collections import defaultdict
import numpy as np
import sys

def main(args):
    relevant_scores = defaultdict(list)
    relevant_score_histogram = defaultdict(int)
    model_scores_by_hit = {'transformer': defaultdict(list), 'bart': defaultdict(list), 'bart_misinfo': defaultdict(list), 'bart_misinfo_pplm': defaultdict(list), 'ref': defaultdict(list)}
    model_scores = {'transformer': [], 'bart': [], 'bart_misinfo': [], 'bart_misinfo_pplm': [], 'ref': []}
    worker_scores = defaultdict(list)
    with open(args.mturk_batch1_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.relevant_%d.%d' % (i, i)] == 'true':
                    relevant_scores[hit_id].append(i)
                    relevant_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)

    with open(args.mturk_batch2_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            for i in range(2):
                if row['Answer.relevant_%d.%d' % (i, i)] == 'true':
                    relevant_scores[hit_id].append(i)
                    relevant_score_histogram[i] += 1
                    model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)


    for model in model_scores_by_hit:
        for hit_id in model_scores_by_hit[model]:
            # print(model_scores_by_hit[model][hit_id])
            if model_scores_by_hit[model][hit_id].count(1) > model_scores_by_hit[model][hit_id].count(0):
                model_scores[model].append(1)
            else:
                model_scores[model].append(0)

    # print('\n\n')
    relevant_score_var = 0
    relevant_agreement = 0
    for hit_id in relevant_scores:
        # print(relevant_scores[hit_id])
        relevant_score_var += np.var(relevant_scores[hit_id])
        for i in range(2):
            if relevant_scores[hit_id].count(i) > 1:
            # if relevant_scores[hit_id].count(i) > 2 or (relevant_scores[hit_id].count(i) + relevant_scores[hit_id].count(i-1)) > 2 or (relevant_scores[hit_id].count(i) + relevant_scores[hit_id].count(i+1)) > 2:
                relevant_agreement += 1
                break

    print('relevant score variance %.2f' % (relevant_score_var*1./len(relevant_scores)))
    print('relevant agreement: %d out of %d' % (relevant_agreement, len(relevant_scores)))

    for model in model_scores:
        print('%s: %.2f' % (model, sum(model_scores[model])*1.0/len(model_scores[model])))

    print('relevant score histogram')
    print(relevant_score_histogram)

    # for worker_id in worker_scores:
    #     print(worker_id)
    #     print(worker_scores[worker_id])
    #     print(worker_scores[worker_id].count(2))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mturk_batch1_results_file", type = str)
    argparser.add_argument("--mturk_batch2_results_file", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)