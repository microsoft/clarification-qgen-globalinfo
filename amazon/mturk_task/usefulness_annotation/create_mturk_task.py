import argparse
import csv
import json
import random
import sys

def count_words(sents):
    ct = 0 
    for sent in sents:
        ct += len(sent.split())
    return ct

def main(args):
    output_data = []
    total = 0
    ignored = 0
    with open(args.json_data_file) as json_file:
        data = json.load(json_file)
        for prod_id in data:
            title = data[prod_id]['title']
            category = data[prod_id]['category']
            description_sents = [sent for sent in data[prod_id]['description'] if 'font-family' not in sent]
            word_count = count_words(description_sents)
            total += 1
            if word_count > 200 or word_count < 10:
                ignored += 1
                continue
            metadata = []
            if data[prod_id]['table1'] != "":
                for key, val in data[prod_id]['table1'].items():
                    metadata.append("%s: %s" % (key, val))
            if data[prod_id]['table2'] != "":
                for key, val in data[prod_id]['table2'].items():
                    metadata.append("%s: %s" % (key, val))
            for question in data[prod_id]['questions']:
                output_data.append([title, category,'\n'.join(description_sents), '\n'.join(metadata), question['question']])
    print('Ignored %d out of %d' % (ignored, total))
    random.shuffle(output_data)
    
    with open(args.output_csv_file_prefix+'_batch1.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['title', 'category', 'description', 'metadata', 'question'])
        for row in output_data[:1000]:
            csv_writer.writerow(row)
    with open(args.output_csv_file_prefix+'_batch2.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['title', 'category', 'description', 'metadata', 'question'])
        for row in output_data[1000:2000]:
            csv_writer.writerow(row)
    with open(args.output_csv_file_prefix+'_batch3.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['title', 'category', 'description', 'metadata', 'question'])
        for row in output_data[2000:3000]:
            csv_writer.writerow(row)
    with open(args.output_csv_file_prefix+'_batch4.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['title', 'category', 'description', 'metadata', 'question'])
        for row in output_data[3000:4000]:
            csv_writer.writerow(row)
    with open(args.output_csv_file_prefix+'_batch5.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['title', 'category', 'description', 'metadata', 'question'])
        for row in output_data[4000:]:
            csv_writer.writerow(row)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--json_data_file", type = str)
    argparser.add_argument("--output_csv_file_prefix", type = str)
    args = argparser.parse_args()
    print(args)
    main(args)