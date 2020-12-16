import torch
import os
import numpy as np
from tqdm.std import tqdm
from sentence_transformers import SentenceTransformer
from data import load_model, load_open_word_test, target_filter, load_entities, relation_tail_train
from fcn_model import mapper


def score_transe(h,r,t):
    return np.linalg.norm(h+r-t, ord=1)

def score_complex(h_r, h_i, t_r, t_i, r_r, r_i):
    return -np.sum(h_r * t_r * r_r
            + h_i * t_i * r_r
            + h_r * t_i * r_i
            - h_i * t_r * r_i,
            -1)

def run_tester(single=False, file=None, model_str='transe'):
    if model_str == 'transe':
        model_path = './checkpoint/mapper.pt'
        if not os.path.exists(model_path):
            exit("Train mapper first!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = mapper().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        filter = target_filter()

        if single:
            with open(file,'r') as f:
                data = f.readlines()
                usr_r = data[0].strip()
                text = data[1].strip()
            
            nlp = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
            this_x = nlp.encode(text)
            output = model(torch.from_numpy(this_x).float().to(device))
            output = output.detach().cpu().numpy()

            e,r = load_model(model_str)

            i_j = {}
            for j, ee in enumerate(e):
                d = score_transe(output, r[int(usr_r)], ee)      
                i_j[j] = d

            sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

            entities = load_entities()
            for i in list(sorted_d)[:10]:
                print(entities[str(i)])


        else:
            hs, ts, rs, hs_name = load_open_word_test(device, deep_filtered=True)

            r_t = relation_tail_train()

            count_1 = 0
            count_3 = 0
            count_10 = 0

            print('Testing...')

            for i in tqdm(range(len(hs))):
                this_desc_embedding = hs[i]
                
                output = model(torch.from_numpy(this_desc_embedding).float().to(device))
                output = output.detach().cpu().numpy()

                e,r = load_model(model_str)

                gt = ts[i]
                this_relation = rs[i]
                this_name = hs_name[i]

                this_filter = []
                for tail in tuple(filter[this_name+':'+this_relation]):
                    if tail != gt:
                        this_filter.append(tail)

                i_j = {}
                for j, ee in enumerate(e):
                    d = score_transe(output, r[int(this_relation)], ee)
                    if str(j) not in this_filter:
                        i_j[str(j)] = d


                sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

                if gt in list(sorted_d)[:10]:
                    count_10 += 1
                if gt in list(sorted_d)[:3]:
                    count_3 += 1
                if gt in list(sorted_d)[:1]:
                    count_1 += 1
                
            print('hits@1: %.1f' % (count_1/len(hs)*100))
            print('hits@3: %.1f' % (count_3/len(hs)*100))
            print('hits@10: %.1f' % (count_10/len(hs)*100))
    
    elif model_str == 'complex':
        model_r_path = './checkpoint/mapper_complex_r.pt'
        model_i_path = './checkpoint/mapper_complex_i.pt'
        if not os.path.exists(model_r_path) and not not os.path.exists(model_i_path):
            exit("Train mapper first!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_r = mapper().to(device)
        model_i = mapper().to(device)
        model_r.load_state_dict(torch.load(model_r_path))
        model_i.load_state_dict(torch.load(model_i_path))
        model_r.eval()
        model_i.eval()
        

        if single:
            with open(file,'r') as f:
                data = f.readlines()
                usr_r = int(data[0].strip())
                text = data[1].strip()
            
            nlp = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device)
            this_x = nlp.encode(text)
            output_r = model_r(torch.from_numpy(this_x).float().to(device))
            output_i = model_i(torch.from_numpy(this_x).float().to(device))
            output_r = output_r.detach().cpu().numpy()
            output_i = output_i.detach().cpu().numpy()
            e,r = load_model(model_str)
            e_r = e[0]
            e_i = e[1]
            r_r = r[0]
            r_i = r[1]

            i_j = {}
            for j in range(len(e_r)):
                d = score_complex(output_r, output_i, e_r[j], e_i[j], r_r[usr_r], r_i[usr_r])
                i_j[j] = d

            sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

            entities = load_entities()
            for i in list(sorted_d)[:10]:
                print(entities[str(i)])

        else:
            hs, ts, rs, _ = load_open_word_test(device, deep_filtered=True)

            count_1 = 0
            count_3 = 0
            count_10 = 0

            print('Testing...')

            filter = target_filter()

            for i in tqdm(range(len(hs))):
                this_desc_embedding = hs[i]
                
                output_r = model_r(torch.from_numpy(this_desc_embedding).float().to(device))
                output_i = model_i(torch.from_numpy(this_desc_embedding).float().to(device))
                output_r = output_r.detach().cpu().numpy()
                output_i = output_i.detach().cpu().numpy()

                e,r = load_model(model_str)
                e_r = e[0]
                e_i = e[1]
                r_r = r[0]
                r_i = r[1]

                gt = ts[i]
                this_relation = rs[i]
                this_filter = []
                for tail in tuple(filter[this_relation]):
                    if tail != gt:
                        this_filter.append(tail)

                i_j = {}
                for j in range(len(e_r)):
                    d = score_complex(output_r, output_i, e_r[j], e_i[j], r_r[int(this_relation)], r_i[int(this_relation)])
                    if str(j) not in this_filter:
                        i_j[str(j)] = d

                sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

                if gt in list(sorted_d)[:10]:
                    count_10 += 1
                if gt in list(sorted_d)[:3]:
                    count_3 += 1
                if gt in list(sorted_d)[:1]:
                    count_1 += 1
                
            print('hits@1: %.1f' % (count_1/len(hs)*100))
            print('hits@3: %.1f' % (count_3/len(hs)*100))
            print('hits@10: %.1f' % (count_10/len(hs)*100))

    
# run_tester(model_str='transe')







