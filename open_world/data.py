import pickle
import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
# from name_entity_recognition import spacy_filter


def load_model(model='transe'):
    if model == 'transe':
        with open('./openke_models/transe300/entities.p', 'rb') as handle:
            e = pickle.load(handle)
        with open('./openke_models/transe300/relations.p', 'rb') as handle:
            r = pickle.load(handle)
    elif model == 'complex':
        with open('./openke_models/complex300/entities_r.p', 'rb') as handle:
            e_r = pickle.load(handle)
        with open('./openke_models/complex300/entities_i.p', 'rb') as handle:
            e_i = pickle.load(handle)
        e = [e_r, e_i]
        with open('./openke_models/complex300/relations_r.p', 'rb') as handle:
            r_r = pickle.load(handle)
        with open('./openke_models/complex300/relations_i.p', 'rb') as handle:
            r_i = pickle.load(handle)
        r = [r_r, r_i]
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
    # print("Number of entities: %d." % (len(entities)))
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
    # print("Number of entities: %d." % (len(relations)))
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
    # print("Number of desc: %d." % (len(descriptions)))
    return descriptions


def load_data(device, cut=20000):
    e,_ = load_model()
    entities = load_entities()
    descriptions = load_descriptions()
    
    print('device: %s' % device)
    # model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device=device)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    print("Transforming descriptions using sbert...")

    if not os.path.exists('./npdave/x.npy') or not os.path.exists('./npdave/y.npy'):
        x = []
        y = []
        for i in tqdm(range(len(e))):
            _id = str(i)
            name = entities[_id]
            this_desc = descriptions[name]
            this_embedding = e[i]
            x.append(model.encode(this_desc))
            y.append(np.array(this_embedding))

        x = np.array([i for i in x]).astype(np.float)
        y = np.array([i for i in y]).astype(np.float)

        np.save('./npdave/x.npy',x)
        np.save('./npsave/y.npy',x)
        print("Transformation done!")
    
    else:
        x = np.load('./npdave/x.npy')
        y = np.load('./npdave/y.npy')

    return x[:cut], y[:cut], x[cut:], y[cut:]
        

def load_data_complex(device, cut=20000):
    e,_ = load_model('complex')
    entities = load_entities()
    descriptions = load_descriptions()
    
    print('device: %s' % device)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    print("Transforming descriptions using sbert...")

    x = []
    yr = []
    yi = []
    for i in tqdm(range(len(e[0]))):
        _id = str(i)
        name = entities[_id]
        this_desc = descriptions[name]
        this_embedding_r = e[0][i]
        this_embedding_i = e[1][i]
        x.append(model.encode(this_desc))
        yr.append(np.array(this_embedding_r))
        yi.append(np.array(this_embedding_i))
    x = np.array([i for i in x]).astype(np.float)
    yr = np.array([i for i in yr]).astype(np.float)
    yi = np.array([i for i in yi]).astype(np.float)
    print("Transformation done!")
    return x[:cut], yr[:cut], yi[:cut], x[cut:], yr[cut:], yi[cut:]

def relation_tail_train():
    r_t = []
    with open('./dbpedia50/train2id.txt', 'r') as ft:
        data = ft.readlines()
        for line in data:
            e = line.strip().split('\t')
            # h = e[0].strip()
            t = e[1].strip()
            r = e[2].strip()
            r_t.append(r+':'+t)
    return tuple(r_t)

# def filter_open_word_test(deep_filtered=False):
#     entities = load_entities()
#     relations = load_relations()
#     if deep_filtered:
#         r_t = relation_tail_train()
#     else:
#         r_t = ()
#     with open('./open_world_dbpedia50/test_tail_open_converted.txt','r') as ef:
#         data = ef.readlines()
#         with open('./open_world_dbpedia50/test_tail_open_converted_filtered.txt', 'w') as wf:       
#             for line in data:
#                 e = line.strip().split('\t')
#                 h = e[0].strip()
#                 t = e[1].strip()
#                 r = e[2].strip()
#                 if deep_filtered:
#                     if r+':'+t in r_t:
#                         wf.write(h+'\t'+t+'\t'+r+'\n')
#                 else:
#                     if t in entities.values() and r in relations.values():
#                         wf.write(h+'\t'+t+'\t'+r+'\n')

def load_open_word_test(device, deep_filtered=False):
    # if not os.path.exists('./open_world_dbpedia50/test_filtered.txt'):
    #     print('Filtering test file...')
    #     filter_open_word_test(deep_filtered)

    entities  = dict((v,k) for k,v in load_entities().items())
    relations = dict((v,k) for k,v in load_relations().items())
    descs = load_descriptions()
    # model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device=device)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)

    hs_name = []
    hs = []
    ts = []
    rs = []

    print("Transforming descriptions using sbert...")
    with open('./open_world_dbpedia50/test_tail_open_converted.txt', 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            e = line.strip().split('\t')
            h = e[0].strip()
            t = e[1].strip()
            r = e[2].strip()

            this_desc = descs[h]
            embeddings = model.encode(this_desc)
            hs.append(embeddings)
            hs_name.append(h)
            t_id = entities[t]
            ts.append(t_id)
            r_id = relations[r]
            rs.append(r_id)

    print("Transformation done!")
    return hs, ts, rs, hs_name

def target_filter():
    filter_list = {}
    relations = dict((v,k) for k,v in load_relations().items())
    entities  = dict((v,k) for k,v in load_entities().items())
    with open('./open_world_dbpedia50/test_tail_open_converted.txt', 'r') as ft:
        data = ft.readlines()
        for line in data:
            e = line.strip().split('\t')
            h = e[0].strip()
            t = e[1].strip()
            r = e[2].strip()
            r = relations[r]
            t = entities[t]
            if h+':'+r not in filter_list:
                filter_list[h+':'+r] = [t]
            else:
                this_list = filter_list[h+':'+r]
                this_list.append(t)
                filter_list[h+':'+r] = this_list
        
    return filter_list

def load_data_for_bert(cut=20000):
    e,_ = load_model()
    entities = load_entities()
    descriptions = load_descriptions()

    x = []
    y = []
    for i in range(len(e)):
        _id = str(i)
        name = entities[_id]
        this_desc = descriptions[name][:500] #cut description, else, too intensive for local hardware
        this_embedding = e[i]
        x.append(this_desc)
        y.append(np.array(this_embedding))

    y = np.array([i for i in y]).astype(np.float)

    return x[:cut], y[:cut], x[cut:], y[cut:]

