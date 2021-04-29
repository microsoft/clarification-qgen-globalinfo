import argparse
import csv
from collections import defaultdict
import numpy as np
import sys

def main(args):
    missing_info_scores = defaultdict(list)
    usefulness_scores = defaultdict(list)
    missing_info_score_histogram = defaultdict(int)
    usefulness_score_histogram = defaultdict(int)

    with open(args.mturk_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            for i in range(1, 6):
                if row['Answer.missing_info_%d.%d' %(i, i)] == 'true':
                    # import pdb; pdb.set_trace()
                    missing_info_scores[hit_id].append(i)
                    missing_info_score_histogram[i] += 1
            for i in range(6):
                if row['Answer.usefulness_%d.%d' % (i, i)] == 'true':
                    usefulness_scores[hit_id].append(i)
                    usefulness_score_histogram[i] += 1

    missing_info_score_var = 0
    usefulness_score_var = 0
    missing_info_agreement = 0
    usefulness_agreement = 0
    for hit_id in usefulness_scores:
        missing_info_score_var += np.var(missing_info_scores[hit_id])
        usefulness_score_var += np.var(usefulness_scores[hit_id])
        for i in range(1, 6):
            if missing_info_scores[hit_id].count(i) > 2:
                missing_info_agreement += 1
                break
        for i in range(6):
            if usefulness_scores[hit_id].count(i) > 1:
            # if usefulness_scores[hit_id].count(i) > 2 or (usefulness_scores[hit_id].count(i) + usefulness_scores[hit_id].count(i-1)) > 2 or (usefulness_scores[hit_id].count(i) + usefulness_scores[hit_id].count(i+1)) > 2:
                usefulness_agreement += 1
                break

    print('Missing info score variance %.2f' % (missing_info_score_var*1./len(missing_info_scores)))
    print('Usefulness score variance %.2f' % (usefulness_score_var*1./len(usefulness_scores)))
    print('Missing info agreement: %d out of %d' % (missing_info_agreement, len(missing_info_scores)))
    print('Usefulness agreement: %d out of %d' % (usefulness_agreement, len(usefulness_scores)))

    print('Missing info score histogram')
    print(missing_info_score_histogram)
    print('Usefulness score histogram')
    print(usefulness_score_histogram)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mturk_results_file", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)