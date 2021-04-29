import argparse
import ast
import csv
import random
import re
import sys
csv.field_size_limit(sys.maxsize)

def clean_html(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return text

def main(args):
    title_dict = {}
    category_dict = {}
    description_dict = {}
    gan_outputs = {}
    gan_asins = [line.strip('\n') for line in open(args.gan_asins_file, 'r').readlines()]
    i = 0
    for line in open(args.gan_predictions_file, 'r').readlines():
        asin = gan_asins[i]
        i += 1
        gan_outputs[asin] = line.strip('\n').replace(' <EOS>', '?')
    model_outputs = []
    with open(args.input_csv_file) as input_csv_file:
        csv_reader = csv.DictReader(input_csv_file, delimiter=',')
        for row in csv_reader:            
            asin = row['asin']
            if asin == 'asin':
                continue
            if asin == '':
                continue
            category = row['assigned_cluster'].replace('&amp;', '&')
            category_dict[asin] = category
            title = row['title'].replace('&amp;', '&')
            title_dict[asin] = title
            description = ' '.join(ast.literal_eval(row['description']))
            description = description.replace('\n', '. ')
            description = clean_html(description)
            description_dict[asin] = description
            ref_question = row['Gold question']
            model_outputs.append([asin, ref_question, 'ref'])
            gan_question = gan_outputs[asin]
            model_outputs.append([asin, gan_question, 'gan'])
            bart_question = row['Baseline-BART']
            model_outputs.append([asin, bart_question, 'bart'])
            bart_misinfo_question = row['BART + Missing Info']
            model_outputs.append([asin, bart_misinfo_question, 'bart_misinfo'])
            bart_misinfo_pplm_question = row['BART + Missing Info + PPLM']
            model_outputs.append([asin, bart_misinfo_pplm_question, 'bart_misinfo_pplm'])

    # with open(args.pilot_input_csv_file) as pilot_input_csv_file:
    #     csv_reader = csv.DictReader(pilot_input_csv_file, delimiter=',')
    #     for row in csv_reader:            
    #         asin = row['asin']
    #         if asin == 'asin':
    #             continue
    #         if asin == '':
    #             continue
    #         category = row['assigned_cluster'].replace('&amp;', '&')
    #         category_dict[asin] = category
    #         title = row['title'].replace('&amp;', '&')
    #         title_dict[asin] = title
    #         description = ' '.join(ast.literal_eval(row['description']))
    #         description = description.replace('\n', '. ')
    #         description = clean_html(description)
    #         description_dict[asin] = description
    #         gan_question = gan_outputs[asin]
    #         model_outputs.append([asin, gan_question, 'gan'])

    random.shuffle(model_outputs)
    with open(args.output_csv_file, 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',')
        csv_writer.writerow(['asin', 'title', 'category', 'description', 'question', 'model'])
        for model_output in model_outputs:
            asin, question, model = model_output
            csv_writer.writerow([asin, title_dict[asin], category_dict[asin], description_dict[asin], question, model])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--input_csv_file', type = str)
    # argparser.add_argument('--pilot_input_csv_file', type = str)
    argparser.add_argument('--output_csv_file', type = str)
    argparser.add_argument('--gan_predictions_file', type = str)
    argparser.add_argument('--gan_asins_file', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)