import pickle
from torch._C import device
from tqdm import tqdm
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os


def load_model(model='transe'):
    if model == 'transe':
        with open('./transe300/entities.p', 'rb') as handle:
            e = pickle.load(handle)
        with open('./transe300/relations.p', 'rb') as handle:
            r = pickle.load(handle)
    else:
        exit('Not support yet!')
    return e, r

def load_entities():
    entities = {}
    with open('./dbpedia50/entity2id.txt','r') as ef:
        data = ef.readlines()[1:]
        for line in data:
            e = line.strip().split('\t')
            name = e[0].strip()
            _id = e[1].strip()
            entities[_id] = name
    print("Number of entities: %d." % (len(entities)))
    return entities

def load_relations():
    relations = {}
    with open('./dbpedia50/relation2id.txt','r') as ef:
        data = ef.readlines()[1:]
        for line in data:
            e = line.strip().split('\t')
            name = e[0].strip()
            _id = e[1].strip()
            relations[_id] = name
    print("Number of entities: %d." % (len(relations)))
    return relations

def load_descriptions():
    descriptions = {}
    with open('./dbpedia50/descriptions.txt','r') as f:
        data = f.readlines()
        for line in data:
            e = line.strip().split('\t')
            name = e[0].strip()
            desc = e[2].strip()
            descriptions[name] = desc
    print("Number of desc: %d." % (len(descriptions)))
    return descriptions


def load_data(device, cut=20000):
    e,_ = load_model()
    entities = load_entities()
    descriptions = load_descriptions()
    x = []
    y = []
    print('device: %s' % device)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    print("Transforming descriptions using sbert...")
    for i in tqdm(range(len(e))):
        _id = str(i)
        name = entities[_id]
        this_desc = descriptions[name]
        this_embedding = e[i]
        x.append(model.encode(name+' '+this_desc))
        y.append(np.array(this_embedding))

    x = np.array([i for i in x]).astype(np.float)
    y = np.array([i for i in y]).astype(np.float)
    print("Transformation done!")
    return x[:cut], y[:cut], x[cut:], y[cut:]

def filter_open_word_test():
    # entities_train = []
    # relations_train = []
    # with open('./dbpedia50/train.txt', 'r') as trf:
    #     data = trf.readlines()
    #     for line in data:
    #         e = line.strip().split('\t')
    #         h = e[0].strip()
    #         t = e[1].strip()
    #         r = e[2].strip()
    #         entities_train.append(h)
    #         entities_train.append(t)
    #         relations_train.append(r)
    # entities_train = tuple(entities_train)
    # relations_train = tuple(relations_train)

    entities = load_entities()
    relations = load_relations()
    with open('./open_world_dbpedia50/test.txt','r') as ef:
        data = ef.readlines()
        with open('./open_world_dbpedia50/test_filtered.txt', 'w') as wf:       
            for line in data:
                e = line.strip().split('\t')
                h = e[0].strip()
                t = e[1].strip()
                r = e[2].strip()
                if t in entities.values() and r in relations.values():
                    wf.write(h+'\t'+t+'\t'+r+'\n')

def load_open_word_test(device):
    if not os.path.exists('./open_world_dbpedia50/test_filtered.txt'):
        print('Filtering test file...')
        filter_open_word_test()

    entities  = dict((v,k) for k,v in load_entities().items())
    relations = dict((v,k) for k,v in load_relations().items())
    descs = load_descriptions()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)

    hs = []
    ts = []
    rs = []

    print("Transforming descriptions using sbert...")
    with open('./open_world_dbpedia50/test_filtered.txt', 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            e = line.strip().split('\t')
            h = e[0].strip()
            t = e[1].strip()
            r = e[2].strip()
            this_desc = descs[h]
            embeddings = model.encode(this_desc)
            hs.append(embeddings)
            t_id = entities[t]
            ts.append(t_id)
            r_id = relations[r]
            rs.append(r_id)
    print("Transformation done!")
    return hs, ts, rs
