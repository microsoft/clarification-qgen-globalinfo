import networkx as nx
import yake
# import stanfordnlp
import stanza
import itertools

# build pipelines
# nlp_pipeline = stanza.Pipeline('en')
# kw_extractor = yake.KeywordExtractor(n=2)

def build_schema(text, nlp_pipeline, kw_extractor):

    '''
    example usage:
    text = 'Is it possible to read using this product at night?'
    nlp_pipeline = stanfordnlp.Pipeline()
    kw_extractor = yake.KeywordExtractor(n=2)

    build_schema(text, nlp_pipeline, kw_extractor)
    '''

    # run a sent tokenizer

    # run the following for each sent

    # VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    VERB_TAGS = ['VB', 'VBG', 'VBZ']

    # run StanfordNLP pipeline
    doc = nlp_pipeline(text)
    sent  = doc.sentences[0]

    # all tokens + root
    tokens = ['root']
    tokens += [t.words[0].text.lower() for t in sent.tokens]

    # obtain all verbs and their indices
    verbs = []
    verb_indices = []
    for t in sent.tokens:
        if t.words[0].xpos in VERB_TAGS:
            verbs.append(t.words[0].text)
            verb_indices.append(int(t.words[0].id))

    # remove all verbs from keywords
    try:
        keywords = kw_extractor.extract_keywords(text)
    except Exception:
        keywords = []

    unigram_keywords_map = {}
    for kw in keywords:
        unigram_keywords_map[kw[0].lower()] = kw[0].lower().split()
    
    unigram_keywords = list(itertools.chain(*list(unigram_keywords_map.values())))

    # after this everything will be unigrams

    keywords_wo_verbs = [kw for kw in unigram_keywords if kw not in verbs]

    # obtain all keyword indices
    keyword_indices = [tokens.index(kw) for kw in keywords_wo_verbs if kw in tokens]

    # initialize dependency tree
    G = nx.DiGraph()
    for dep_edge in sent.dependencies:
        G.add_edge(int(dep_edge[0].id), int(dep_edge[2].id), relation=dep_edge[1])

    relation_edge_dict = nx.get_edge_attributes(G,'relation')

    schema = {}

    tuple_schema = []

    for v in verb_indices:
        # find a path from verb v to each keywords
        for k in keyword_indices:
            # we are finding shortest paths
            try:
                path = nx.shortest_path(G, source=v, target=k)
            except:
                # print('No path obtained')
                continue
                # check is path contains more than 2 nodes
            if len(path) > 2:
                # walk backward from the target to source
                for i, node in reversed(list(enumerate(path))):
                    if node in verb_indices:
                        # retrieve the first parent verb of the keyword
                        # obtain the relation with its child on the path
                        tuple_schema.append((tokens[v], tokens[k], relation_edge_dict[(node, path[i+1])], len(path)))
                        break
            else:
                # default case for an one-hop path between verb and keyword
                tuple_schema.append((tokens[v], tokens[k], relation_edge_dict[(v, k)], 2))

    # retain only the closest verb for each keyword
    for kw in keywords_wo_verbs:
        kw_tuples = [t for t in tuple_schema if t[1] == kw]
        if kw_tuples:
            final_tuple = min(kw_tuples, key = lambda t: t[1])  
            schema[kw] = final_tuple[:-1] # drop the path length
        else:
            schema[kw] = kw

    merged_schema = {}
    captured_uni_kw = []
    for kw, uni_kw in unigram_keywords_map.items():
        if uni_kw[0] in keywords_wo_verbs:
            if uni_kw[0] not in captured_uni_kw:
                # collect tuples based on the first unigram entry
                merged_schema[kw] = schema[uni_kw[0]]
                captured_uni_kw.extend(uni_kw)
    
    for kw, t in merged_schema.items():
        if isinstance(t, tuple):  
            merged_schema[kw] = (t[0], kw, t[-1])
        else:
            merged_schema[kw] = kw

    merged_schema = list(merged_schema.values())

    # default case, no schema, hence keep all keywords including verbs
    if len(tuple_schema) == 0:
        merged_schema = [kw[0].lower() for kw in keywords]

    merged_schema = list(set(merged_schema))

    return merged_schema

# build schema for product descriptions

def build_schema_from_desc(text, kw_extractor):
    # remove all verbs from keywords
    try:
        keywords = kw_extractor.extract_keywords(text)
    except Exception:
        keywords = []

    unigram_keywords_map = {}
    for kw in keywords:
        unigram_keywords_map[kw[0].lower()] = kw[0].lower().split()

    merged_schema = []

    captured_uni_kw = []
    for kw, uni_kw in unigram_keywords_map.items():
        if uni_kw[0] not in captured_uni_kw:
            # collect tuples based on the first unigram entry
            merged_schema.append(kw)
            captured_uni_kw.extend(uni_kw)
    
    return merged_schema

def build_schema_from_table(table):
    if table:
        return list(table.keys())
    else:
        return []


if __name__ == "__main__":

    from urllib.request import urlopen
    from collections import defaultdict
    from tqdm import tqdm

    import os
    import json
    import gzip
    import pickle
    import pandas as pd
 
    ### load the meta data

    data = []
    with gzip.open('meta_Electronics.json.gz') as f:
            for l in f:
                data.append(json.loads(l.strip()))

    df = pd.DataFrame.from_dict(data)
    df3 = df.fillna('')
    df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
    df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows

    ### load the QA

    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    qa_df = getDF('qa_Electronics.json.gz')

    final_df = pd.merge(df5, qa_df, on='asin')
    final_df = final_df.drop_duplicates(subset=['question'])
    print('Overlap for {} items'.format(len(final_df)))

    only_product_df = final_df.drop_duplicates(subset=['asin'])

    category_dict = defaultdict(lambda: defaultdict(int))
    for i, row in tqdm(only_product_df.iterrows()):
        category = row['category']
        for i, c in enumerate(category):
            category_dict[i][c] += 1

    print('Number of categories: {}'.format(len(category_dict)))

    #assign clusters

    assigned_cluster = []
    assigned_cluster_level = []
    for i, row in tqdm(final_df.iterrows()):
        category = row['category']
        for i, c in enumerate(category):
            if category_dict[i][c] < 400:
                assigned_cluster.append(c)
                assigned_cluster_level.append(i)
                break
            elif i == len(category)-1:            
                assigned_cluster.append(c)
                assigned_cluster_level.append(i)


    final_df['assigned_cluster'] = assigned_cluster
    final_df['assigned_cluster_level'] = assigned_cluster_level
    # print(final_df.head(2)[['category', 'assigned_cluster']])


    # get the schema

    # build pipelines
    nlp_pipeline = stanza.Pipeline()
    kw_extractor = yake.KeywordExtractor(n=2)

    save_to_file = False

    # get global schema for a specific cluster/category
    prod_dict = {}
    cat = "Laptops" # get cluster names from category_dict
    cat_df = final_df[final_df['assigned_cluster'] == cat]
    grps = cat_df.groupby(cat_df['asin'])
    for name, g in grps:
        item_dict = {}
        qa = []
        item_id = g['asin'].iloc[0]
        for i, r in g.iterrows():
            item_dict['title'] = r['title']
            # item_dict['description'] = r['description']
            item_dict['description_schema'] = build_schema_from_desc(' '.join(r['description']), kw_extractor)
            # item_dict['table1'] = r['tech1']
            # item_dict['table2'] = r['tech2']
            table_schema = []
            table_schema.extend(build_schema_from_table(r['tech1']))
            table_schema.extend(build_schema_from_table(r['tech2']))
            item_dict['table_schema'] = table_schema

            schema = build_schema(r['question'].lower(), nlp_pipeline, kw_extractor)
            qa.append({'question': r['question'], 'schema': schema})
        item_dict['questions'] = qa
        
        prod_dict[item_id] = item_dict

    print('Category {} has {} items.'.format(cat, len(prod_dict)))

    if save_to_file:
        with open('{}_schema.json'.format(cat), 'w') as fp:
            json.dump(prod_dict, fp, indent=4)
   