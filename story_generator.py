import pandas as pd
from IPython.display import display, HTML
import gc
import random
import sys
from functools import reduce

task = sys.argv[1]

print("relation:", task)
dataset = sys.argv[2]
kinship="kinship"
fb="freebase"
nell="nell"
countries='countries'

main_dir = '/home/ailab/memory_attention_model/'

task_dir = ''

if fb == dataset:
    task_dir = main_dir + 'data/FB15k-237/tasks/' + task     
elif nell == dataset:
    task_dir = main_dir + 'data/NELL-995/tasks/'
elif kinship == dataset:
    task_dir = main_dir + 'data/kinship/tasks/' + task
elif countries == dataset:
    task_dir = main_dir + 'data/Countries/tasks/' + task

ontology_filepath_nell = main_dir + 'data/NELL-995/nell_ontology.csv'
ontology_filepath_fb   = main_dir + 'data/FB15k-237/fb15k_ontology.txt'
ontology_filepath_kinship = main_dir
ontology_filepath_countries = main_dir + 'data/Countries/country_schema.txt'

if fb == dataset:
    task_uri = '/' + task.replace('@', '/')
elif nell == dataset or kinship == dataset:
    task_uri = task
elif countries == dataset:
    task_uri = 'locatedIn'

def clean_nell(line):
    return line.replace('\n', '').replace('concept:', '').replace('thing$', '').replace("concept_", '').replace(":", "_")
if nell == dataset:
    relation_paths = []
    #with open(main_dir + "data/NELL-995/paths_with_threshold_0.1/" + task) as raw_file: # path length = 3
    with open(main_dir + "data/NELL-995/paths_with_length_of_4_and_threshold_0.5/" + task) as raw_file:
        for line in raw_file.readlines()[1:-2]:
            path = line.replace(":", "_").replace('\n', '').split(',')
            out_path = []
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)
            if 'date' not in path:
                relation_paths.append(path)
                
    kb = list()
    with open('/home/ailab/memory_attention_model/data/NELL-995/tasks/' + 'concept_' + task + "/graph.txt") as graph_file:
        for line in graph_file:
            if 'latitudelongitude' not in line and 'date' not in line:
                e1, r, e2 = clean_nell(line).split('\t')
                kb.append((e1, r, e2))         

    train_data = []
    with open(task_dir + 'concept_' + task + '/train.pairs') as raw_file:
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = e1[6:]
            e2 = e2[6:]     
            train_data.append((e1, e2, label))

    test_data = []
    with open(task_dir + 'concept_' + task + '/sort_test.pairs') as raw_file: 
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = e1[6:]
            e2 = e2[6:]
            test_data.append((e1, e2, label)) 
elif fb == dataset:
    relation_paths = []
    
    #with open(main_dir + "data/FB15k-237/paths/" + task) as raw_file:
    with open(main_dir + 'data/FB15k-237/paths_with_length_4/' + task) as raw_file:
        for line in raw_file.readlines()[3:]:
            path = line.replace("-", "/").replace('\n', '').split(',')
            out_path = []
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)
            if 'date' not in path:
                relation_paths.append(path)
                
    kb = list()
    with open(task_dir + "/graph.txt") as graph_file:
        for line in graph_file:
            if 'latitudelongitude' not in line and 'date' not in line:
                e1, r, e2 = clean_nell(line).split('\t')
                kb.append((e1, r, e2))         

    train_data = []
    with open(task_dir + '/train.pairs') as raw_file:
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = '/m/' + e1[8:]
            e2 = '/m/' + e2[8:]     
            train_data.append((e1, e2, label))

    test_data = []
    with open(task_dir + '/sort_test.pairs') as raw_file: 
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = '/m/' + e1[8:]
            e2 = '/m/' + e2[8:]
            test_data.append((e1, e2, label)) 
            
elif kinship == dataset or countries == dataset:
    relation_paths = []
    if dataset == countries:
        path_dir = main_dir + 'data/Countries/paths/' + task
    with open(path_dir) as raw_file:
        #for line in raw_file.readlines()[1:-1]: # kinship
        for line in raw_file.readlines(): # countries
            path = line.replace('\n', '').split(',')
            out_path = []
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)
            
            relation_paths.append(path)
                
    kb = list()
    with open(task_dir + "/graph.txt") as graph_file:
        for line in graph_file:
            e1, r, e2 = clean_nell(line).split('\t')
            kb.append((e1, r, e2))         

    train_data = []
    with open(task_dir + '/train.pairs') as raw_file:
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.strip()[:-3]
            e1, e2 = ee.split(',')
     
            train_data.append((e1, e2, label))

    test_data = []
    with open(task_dir + '/sort_test.pairs') as raw_file: 
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.strip()[:-3]
            e1, e2 = ee.split(',')

            test_data.append((e1, e2, label)) 

train_pos = set(filter(lambda x:x[2] == '+', train_data))
train_pos = list(map(lambda x:(x[0], task_uri, x[1]), train_pos))

train_pos_inv = list(map(lambda x:(x[2], task_uri + '_inv', x[0]), train_pos))

kb = kb + train_pos
kb = kb + train_pos_inv

print('train_data:', len(train_data))
print('test_data:', len(test_data))
print('kb:', len(set(kb)))
print('paths:', len(relation_paths))

if nell == dataset:
    ontology_file = open(ontology_filepath_nell, 'r').readlines()
elif fb == dataset:
    ontology_file = open(ontology_filepath_fb, 'r').readlines()

def get_domrange_nell(target):    
    target_domain = ''
    target_range = ''
    for line in ontology_file:
        if 'concept:' + target + '\tdomain\tconcept' in line:
            target_domain = line.split('\t')[-1].replace('\n', '').split(':')[-1]
        if 'concept:' + target + '\trange\tconcept' in line:
            target_range = line.split('\t')[-1].replace('\n', '').split(':')[-1]

    return (target_domain, target_range)

def get_domrange_fb(target):    
    target_domain = ''
    target_range = ''
    for line in ontology_file:
        if target in line:
            _, target_domain, target_range = line.replace('\n', '').replace('.', '').split('\t')

    return (target_domain, target_range)
def get_domrange_kinship(target):
    return ("person", "person")

relation_list = map(lambda x:x.replace('_inv', '').split(','), relation_paths)
relations = set(reduce(lambda x,y: x + y, relation_list))

meta_dict = {}
if nell == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_nell(r)

elif fb == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_fb(r)

elif kinship == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_kinship(r)

elif countries == dataset:
    ontology_file = open(ontology_filepath_countries, 'r').readlines()
    for line in ontology_file:
        e, e_type = line.split('\t')
        meta_dict[e] = e_type.replace('\n', '')

def get_sentences(input_data, task_uri, relation_paths, kb):
    result = list()
    count = 0
    num_relations = len(relation_paths) 
    input_data = list(map(lambda x:x[:2], input_data))

    i1_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o2'])
    i2_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o3'])
    i3_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o1'])
    ii = 0
    for line in relation_paths:
        r_list = line.split(',')
        ii += 1
        if ii % 100 == 0:
            print(ii)
        if len(r_list) == 2:

            r1, r2 = r_list
            r1 = r1.strip()
            r2 = r2.strip()
            tset_r1 = list(filter(lambda x:x[1] == r1,kb))
            tset_r2 = list(filter(lambda x:x[1] == r2,kb))

            df1 = pd.DataFrame.from_records(tset_r1, columns=['s1', 'p1', 'key'])
            df2 = pd.DataFrame.from_records(tset_r2, columns=['p2', 'key', 'o2'])

            mdf = pd.merge(df1, df2, on='key').drop_duplicates()
            mdf2 = pd.merge(mdf, i1_df, on=['s1', 'o2'])        
            mdf2 = mdf2[['s1', 'p1', 'key', 'p2', 'o2']]

            cr = mdf2.values

            for e in cr:
                result.append(list(e))

            del [[df1, df2, mdf, mdf2]]
            gc.collect()

        elif len(r_list) == 3:

            r1, r2, r3 = r_list
            r1 = r1.strip()
            r2 = r2.strip()
            r3 = r3.strip()
            tset_r1 = list(filter(lambda x:x[1] == r1,kb))
            tset_r2 = list(filter(lambda x:x[1] == r2,kb))
            tset_r3 = list(filter(lambda x:x[1] == r3,kb))

            df1 = pd.DataFrame.from_records(tset_r1, columns=['s1', 'p1', 'key1'])
            df2 = pd.DataFrame.from_records(tset_r2, columns=['key1','p2', 'key2'])
            df3 = pd.DataFrame.from_records(tset_r3, columns=['key2', 'p3', 'o3'])

            mdf = pd.merge(df1, df2, on='key1').drop_duplicates()
            mdf2 = pd.merge(mdf, df3, on='key2').drop_duplicates()

            mdf3 = pd.merge(mdf2, i2_df, on=['s1', 'o3'])
            mdf3 = mdf3[['s1', 'p1', 'key1', 'p2', 'key2', 'p3', 'o3']]
            
            cr = mdf3.values

            for e in cr:
                result.append(list(e))

            del [[df1, df2, df3, mdf, mdf2, mdf3]]
            gc.collect() 

        elif len(r_list) == 1 and task_uri != r_list[0]:

            r1 = r_list[0].strip()
            tset_r1 = list(filter(lambda x:x[1] == r1,kb))

            label1 = ['s1', 'p1', 'o1']
            df1 = pd.DataFrame.from_records(tset_r1, columns=label1)
            mdf = pd.merge(df1, i3_df, on=['s1', 'o1'])
            mdf = mdf[['s1', 'p1', 'o1']]
            
            cr = mdf.values

            for e in cr:
                result.append(list(e))

            del [[df1, mdf]]
            gc.collect() 
            
    return result

train_sentences = get_sentences(train_data, task_uri, relation_paths, kb)
test_sentences = get_sentences(test_data, task_uri, relation_paths, kb)

def filter_sent(s):
    #e1 r1 e2 r2 e3 r3 e4
    sent = ' '.join(s)
    triple = s[0] + ' ' + task_uri + ' ' + s[-1]
    inv_triple = s[-1] + ' ' + task_uri + '_inv ' + s[0]
    if triple in sent:
        return False
    if inv_triple in sent:
        return False
    else:
        return True

#print('before sentence filtering:')
#print(len(test_sentences))
#print(len(train_sentences))
#test_sentences = list(filter(filter_sent, test_sentences))
#train_sentences = list(filter(filter_sent, train_sentences))
#print('after sentence filtering:')
#print(len(test_sentences))
#print(len(train_sentences))

def get_dom_range_country(path):
    
    new_path = [''] * len(path)
    
    if len(path) == 7:

        new_path[0] = meta_dict[path[0]]
        new_path[2] = meta_dict[path[2]]
        new_path[4] = meta_dict[path[4]]           
        new_path[6] = meta_dict[path[6]]           
        new_path[1] = path[1]
        new_path[3] = path[3]
        new_path[5] = path[5]
        
    elif len(path) == 5:

        new_path[0] = meta_dict[path[0]]
        new_path[2] = meta_dict[path[2]]
        new_path[4] = meta_dict[path[4]]
        new_path[1] = path[1]
        new_path[3] = path[3]

    else:

        new_path[0] = meta_dict[path[0]]
        new_path[1] = path[1]
        new_path[2] = meta_dict[path[2]]

    #return path
    return new_path

def get_dom_range(path):
    
    new_path = [''] * len(path)

    if nell == dataset:
        new_path[0] = path[0].split('_')[0]
        new_path[-1] = path[-1].split('_')[0]
    
    if len(path) == 7:
        r1 = path[1]
        r2 = path[3]
        r3 = path[5]

        if 'inv' in r1:            
            if fb == dataset or kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]

        if 'inv' in r2:
            new_path[4] = meta_dict[r2.replace('_inv', '')][0]            
        else:
            new_path[4] = meta_dict[r2][1]

        if fb == dataset or kinship:
            if 'inv' in r3:
                new_path[-1] = meta_dict[r3.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r3][1]

        new_path[1] = r1
        new_path[3] = r2
        new_path[5] = r3
        
    elif len(path) == 5:
        r1 = path[1]
        r2 = path[3]
        if 'inv' in r1:            
            if fb == dataset or kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]
        
        if fb == dataset or kinship:
            if 'inv' in r2:
                new_path[-1] = meta_dict[r2.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r2][1]

        new_path[1] = r1
        new_path[3] = r2
    else:
        r1 = path[1]
        if fb == dataset or kinship:
            if 'inv' in r1:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
                new_path[-1] = meta_dict[r1.replace('_inv', '')][0]
            else:
                new_path[0] = meta_dict[r1][0]
                new_path[-1] = meta_dict[r1][1]
        new_path[1] = r1
    return new_path


def make_story(input_sentences, input_data):
    skip_count = 0    
    output_data = []
    output_list = []
    for i, sample in enumerate(input_data):
        e1, e2, l = sample

        cxt_path_list = list(filter(lambda path: path[0] == e1 and path[-1] == e2, input_sentences))
        if dataset == countries:
            cxt_sent = list(set(map(lambda path:" ".join(get_dom_range_country(path)), cxt_path_list))) 
        else:
            cxt_sent = list(set(map(lambda path:" ".join(get_dom_range(path)), cxt_path_list))) 
                
        if len(cxt_sent) != 0:
            context = "\n".join(cxt_sent)
            output_data.append([context, e1, e2, l])
        else:
            skip_count += 1
            
        if i % 1000 == 0:
            print(i, "/", len(input_data))
            
    print("input_data:", len(input_data))
    print('skip_count:', skip_count)
    
    return output_data

train_result = make_story(train_sentences, train_data)
test_result = make_story(test_sentences, test_data)

def convert_to_sentence(input_data):
    output_data = []
    for context, e1, e2, l in input_data:
        #if e1 not in no_queries:
        output_data.append(context + '\n' + e1 + '\t' + task_uri + '\t' + e2 + '\t' + l)
    return output_data

train_out = convert_to_sentence(train_result)
test_out = convert_to_sentence(test_result)
print('num of training samples:', len(train_out))
print('num of test samples:', len(test_out))

import os
# Create output directory
dataDirName = main_dir + "data/processed_data/" + task
 
try:
    os.mkdir(dataDirName)
    print("Directory " , dataDirName, " Created") 
except FileExistsError:
    print("Directory " , dataDirName, " already exists")

with open(dataDirName + "/" + 'train.txt','w') as f:
    f.write( '\n'.join( train_out ) )

with open(dataDirName + "/" + 'test.txt','w') as f:
    f.write( '\n'.join( test_out ) )
