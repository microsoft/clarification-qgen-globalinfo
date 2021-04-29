import argparse
import ast
import csv
import random
import re
import sys
csv.field_size_limit(sys.maxsize)

def format_context(context):
    dialog_context = ''
    prev_user = 'B'
    for line in context:
        if line == '':
            continue
        if prev_user == 'B':
            curr_user = 'A'
        else:
            curr_user = 'B'
        dialog_context += 'User %s: %s<br/>' % (curr_user, line)
        prev_user = curr_user
    if curr_user == 'A':
        next_user = 'B'
    else:
        next_user = 'A'
    return dialog_context, next_user

def main(args):
    context_dict = {}
    ref_question_dict = {}
    model_outputs = []
    ID = 51
    with open(args.input_csv_file) as input_csv_file:
        csv_reader = csv.DictReader(input_csv_file, delimiter=',')
        for row in csv_reader:      
            context = row['\ufeffContext']
            if context == 'Context':
                continue
            if context == '':
                continue
            context_dict[ID], next_user = format_context(context.split('\n'))
            ref_question = row['Gold']
            ref_question_dict[ID] = 'User %s: %s' % (next_user, ref_question)
            transformer_question = row['Transformer']
            bart_question = row['BART-baseline']
            bart_misinfo_question = row['BART-MissingInfo']
            bart_misinfo_pplm_question = row['BART-PPLM']
            model_outputs.append([ID, ('User %s: %s' % (next_user, transformer_question), 'transformer'),
                                        ('User %s: %s' % (next_user, bart_question), 'bart'),
                                        ('User %s: %s' % (next_user, bart_misinfo_question), 'bart_misinfo'),
                                        ('User %s: %s' % (next_user, bart_misinfo_pplm_question), 'bart_misinfo_pplm')])
            ID += 1

    csv_rows = []
    for model_output in model_outputs:
        ID = model_output[0]
        bart_misinfo_pplm_question = model_output[-1][0]
        for i in range(1, 4):
            model_question, model_name = model_output[i]
            swap = random.randint(0, 1)
            if swap:
                model_a_question = bart_misinfo_pplm_question
                model_a_name = 'bart_misinfo_pplm'
                model_b_question = model_question
                model_b_name = model_name
            else:
                model_a_question = model_question
                model_a_name = model_name
                model_b_question = bart_misinfo_pplm_question
                model_b_name = 'bart_misinfo_pplm'
            csv_rows.append([ID, context_dict[ID], ref_question_dict[ID], model_a_question, model_a_name, model_b_question, model_b_name])

    random.shuffle(csv_rows)
    with open(args.output_csv_file, 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',')
        csv_writer.writerow(['ID', 'context', 'ref_question', 'question_a', 'model_a', 'question_b', 'model_b'])
        for csv_row in csv_rows:
            csv_writer.writerow(csv_row)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--input_csv_file', type = str)
    argparser.add_argument('--output_csv_file', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)