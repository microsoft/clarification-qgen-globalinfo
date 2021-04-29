import argparse
import csv
from collections import defaultdict
import numpy as np
import sys

def main(args):
    usefulness_scores = defaultdict(list)
    usefulness_score_histogram = defaultdict(int)
    model_scores_by_hit = {'bart': defaultdict(list), 'bart_misinfo': defaultdict(list), 'bart_misinfo_pplm': defaultdict(list), 'ref': defaultdict(list), 'gan': defaultdict(list)}
    model_scores = {'bart': [], 'bart_misinfo': [], 'bart_misinfo_pplm': [], 'ref': [], 'gan': []}
    worker_scores = defaultdict(list)
    with open(args.mturk_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model = row['Input.model']
            worker_id = row['WorkerId']
            if worker_id in ['A1F1BIPJR11LSR', 'A1MJVTR0PCKBWW']:
                continue
            for i in range(5):
                if row['Answer.usefulness_%d.%d' % (i, i)] == 'true':
                    usefulness_scores[hit_id].append(i)
                    usefulness_score_histogram[i] += 1
                    if i > 1:
                        model_scores_by_hit[model][hit_id].append(1)
                    else:
                        model_scores_by_hit[model][hit_id].append(0)
                    # model_scores_by_hit[model][hit_id].append(i)
                    worker_scores[worker_id].append(i)

    for model in model_scores_by_hit:
        for hit_id in model_scores_by_hit[model]:
            # print(model_scores_by_hit[model][hit_id])
            if model_scores_by_hit[model][hit_id].count(1) > model_scores_by_hit[model][hit_id].count(0):
                model_scores[model].append(1)
            else:
                model_scores[model].append(0)
            # model_scores[model].append(sum(model_scores_by_hit[model][hit_id])*1./len(model_scores_by_hit[model][hit_id]))

    print('\n\n')
    usefulness_score_var = 0
    usefulness_agreement = 0
    for hit_id in usefulness_scores:
        print(usefulness_scores[hit_id])
        usefulness_score_var += np.var(usefulness_scores[hit_id])
        for i in range(5):
            if usefulness_scores[hit_id].count(i) > 1:
            # if usefulness_scores[hit_id].count(i) > 2 or (usefulness_scores[hit_id].count(i) + usefulness_scores[hit_id].count(i-1)) > 2 or (usefulness_scores[hit_id].count(i) + usefulness_scores[hit_id].count(i+1)) > 2:
                usefulness_agreement += 1
                break

    print('Usefulness score variance %.2f' % (usefulness_score_var*1./len(usefulness_scores)))
    print('Usefulness agreement: %d out of %d' % (usefulness_agreement, len(usefulness_scores)))

    for model in model_scores:
        print('%s: %.2f' % (model, sum(model_scores[model])*1.0/len(model_scores[model])))

    print('Usefulness score histogram')
    print(usefulness_score_histogram)

    # for worker_id in worker_scores:
    #     print(worker_id)
    #     print(worker_scores[worker_id])
    #     print(worker_scores[worker_id].count(2))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mturk_results_file", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)