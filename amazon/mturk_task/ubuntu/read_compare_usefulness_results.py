import argparse
import csv
from collections import defaultdict
import sys

model_pair_choices = ['transformer-bart_misinfo_pplm', 'bart-bart_misinfo_pplm', 'bart_misinfo-bart_misinfo_pplm']

def main(args):
    missing_annotations = 0
    model_pair = {}
    model_comparison = defaultdict(list)
    model_comparision_aggr = {'transformer-bart_misinfo_pplm': [0, 0], 'bart-bart_misinfo_pplm': [0, 0], 'bart_misinfo-bart_misinfo_pplm': [0, 0]}
    rejected = 0
    with open(args.mturk_batch1_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model_a = row['Input.model_a']
            model_b = row['Input.model_b']
            if row['RejectionTime'].strip() != '':
                rejected +=1
                continue
            if hit_id not in model_pair:
                if model_a+'-'+model_b in model_pair_choices:
                    model_pair[hit_id] = model_a+'-'+model_b
                elif model_b+'-'+model_a in model_pair_choices:
                    model_pair[hit_id] = model_b+'-'+model_a
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            if row['Answer.usefulness_A.A'] == 'true' :
                if model_a+'-'+model_b == model_pair[hit_id]:
                    model_comparison[hit_id].append('a')
                elif model_b+'-'+model_a == model_pair[hit_id]:
                    model_comparison[hit_id].append('b')
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            elif row['Answer.usefulness_B.B'] == 'true' :
                if model_a+'-'+model_b == model_pair[hit_id]:
                    model_comparison[hit_id].append('b')
                elif model_b+'-'+model_a == model_pair[hit_id]:
                    model_comparison[hit_id].append('a')
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            else:
                missing_annotations += 1

    with open(args.mturk_batch2_results_file) as csv_results_file:
        csv_reader = csv.DictReader(csv_results_file)
        for row in csv_reader:
            hit_id = row['HITId']
            model_a = row['Input.model_a']
            model_b = row['Input.model_b']
            if row['RejectionTime'].strip() != '':
                rejected +=1
                continue
            if hit_id not in model_pair:
                if model_a+'-'+model_b in model_pair_choices:
                    model_pair[hit_id] = model_a+'-'+model_b
                elif model_b+'-'+model_a in model_pair_choices:
                    model_pair[hit_id] = model_b+'-'+model_a
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            if row['Answer.usefulness_A.A'] == 'true' :
                if model_a+'-'+model_b == model_pair[hit_id]:
                    model_comparison[hit_id].append('a')
                elif model_b+'-'+model_a == model_pair[hit_id]:
                    model_comparison[hit_id].append('b')
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            elif row['Answer.usefulness_B.B'] == 'true' :
                if model_a+'-'+model_b == model_pair[hit_id]:
                    model_comparison[hit_id].append('b')
                elif model_b+'-'+model_a == model_pair[hit_id]:
                    model_comparison[hit_id].append('a')
                else:
                    import pdb; pdb.set_trace()
                    assert(False)
            else:
                missing_annotations += 1

    print('Rejected %d' % rejected)
    print('Missing annotations %d' % missing_annotations)
    for hit_id in model_comparison:
        if model_comparison[hit_id].count('a') > 1:
            model_comparision_aggr[model_pair[hit_id]][0] += 1
        elif model_comparison[hit_id].count('b') > 1:
            model_comparision_aggr[model_pair[hit_id]][1] += 1
        else:
            print('no agreement')
            model_comparision_aggr[model_pair[hit_id]][1] += 1

    for curr_model_pair in model_comparision_aggr:
        total = sum(model_comparision_aggr[curr_model_pair])
        print('total %d' % total)
        print(curr_model_pair)
        print('%s wins %.4f' % (curr_model_pair.split('-')[0], model_comparision_aggr[curr_model_pair][0]*1.0/total))
        print('%s wins %.4f' % (curr_model_pair.split('-')[1], model_comparision_aggr[curr_model_pair][1]*1.0/total))
        print('\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mturk_batch1_results_file", type = str)
    argparser.add_argument("--mturk_batch2_results_file", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)