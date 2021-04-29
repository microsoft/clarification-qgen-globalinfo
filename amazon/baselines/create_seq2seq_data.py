import argparse
import ast
import csv
import re
import sys
csv.field_size_limit(sys.maxsize)

def clean_html(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return text

def main(args):
    asin_file = open(args.output_prefix+'_asin.txt', 'w')
    context_file = open(args.output_prefix+'_context.txt', 'w')
    question_file = open(args.output_prefix+'_question.txt', 'w')
    answer_file = open(args.output_prefix+'_answer.txt', 'w')
    with open(args.input_csv_file) as input_csv_file:
        csv_reader = csv.DictReader(input_csv_file, delimiter=',')
        for row in csv_reader:
            asin = row['asin']
            asin_file.write(asin+'\n')
            category = row['assigned_cluster'].replace('&amp;', '&')
            title = row['title'].replace('&amp;', '&')
            description = ' '.join(ast.literal_eval(row['description']))
            description = description.replace('\n', '. ')
            description = clean_html(description)
            question = row['question']
            answer = row['answer']
            context_file.write('. '.join([category, title, description]) + '\n')
            question_file.write(question+'\n')
            answer_file.write(answer+'\n')
    context_file.close()
    question_file.close()
    answer_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--input_csv_file', type = str)
    argparser.add_argument('--output_prefix', type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)