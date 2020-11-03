with open('test_tail_open.txt', 'r') as f:
    data = f.readlines()
    with open('test_tail_open_converted.txt', 'w') as wf:
        for line in data:
            e = line.strip().split('\t')
            h = e[0].strip()
            r = e[1].strip()
            t = e[2].strip()
            wf.write(h+'\t'+t+'\t'+r+'\n')