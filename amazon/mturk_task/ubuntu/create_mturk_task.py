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
            model_outputs.append([ID, 'User %s: %s' % (next_user, ref_question), 'ref'])
            transformer_question = row['Transformer']
            model_outputs.append([ID, 'User %s: %s' % (next_user, transformer_question), 'transformer'])
            bart_question = row['BART-baseline']
            model_outputs.append([ID, 'User %s: %s' % (next_user, bart_question), 'bart'])
            bart_misinfo_question = row['BART-MissingInfo']
            model_outputs.append([ID, 'User %s: %s' % (next_user, bart_misinfo_question), 'bart_misinfo'])
            bart_misinfo_pplm_question = row['BART-PPLM']
            model_outputs.append([ID, 'User %s: %s' % (next_user, bart_misinfo_pplm_question), 'bart_misinfo_pplm'])
            ID += 1

    random.shuffle(model_outputs)
    with open(args.output_csv_file, 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',')
        csv_writer.writerow(['ID', 'context', 'question', 'model'])
        for model_output in model_outputs:
            ID, question, model = model_output
            csv_writer.writerow([ID, context_dict[ID], question, model])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--input_csv_file', type = str)
    argparser.add_argument('--output_csv_file', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)