def translate_files(_from, _to, entities, relations):
    with open(_from,'r') as trf:
        data = trf.readlines()
        with open(_to,'w') as trff:
            length = len(data)
            trff.write(str(length)+'\n')
            for line in data:
                e = line.strip().split('\t')
                h = e[0].strip()
                t = e[1].strip()
                r = e[2].strip()
                trff.write(entities[h]+'\t'+entities[t]+'\t'+relations[r]+'\n')


if __name__ == "__main__":
    entities = {}
    with open('entity2id.txt','r') as ef:
        data = ef.readlines()[1:]
        for line in data:
            e = line.strip().split('\t')
            name = e[0].strip()
            _id = e[1].strip()
            entities[name] = _id
    print(len(entities))

    relations = {}
    with open('relation2id.txt','r') as rf:
        data = rf.readlines()[1:]
        for line in data:
            e = line.strip().split('\t')
            r = e[0].strip()
            _id = e[1].strip()
            relations[r] = _id
    print(len(relations))

    # _from = ['train.txt', 'valid.txt', 'test.txt']
    # _to = ['train2id.txt', 'valid2id.txt', '../kb2E/test2id.txt']
    # for i in range(3):
    #     translate_files(_from[i], _to[i], entities, relations)

