import itertools

alignment_file = 'data/alignment.txt'
item_file = 'data/items.item'

with open(alignment_file) as fin, open(item_file, 'w') as fout:
    fout.write(' '.join(('#file', 'onset', 'offset', '#phone', 'context')))
    fout.write('\n')
    for line in fin:
        splitted = line.split()
        fname = splitted[0]
        phones = map(lambda x: x.split('_')[0], splitted[1:])
        groups = [(phn, len(list(g))) for phn, g in itertools.groupby(phones)]

        # Getting times, assuming frate=100
        time = 0.005  # initial time
        trans = []  # 
        for phn, n in groups:
            trans.append((phn, time, time+n*0.01))
            time += n*0.01

        for prev, curr, nexT in zip(trans[:-2], trans[1:-1], trans[2:]):
            fout.write(' '.join((
                fname, str(curr[1]), str(curr[2]),    # fname, onset, offset
                curr[0], '-'.join((prev[0], nexT[0])),  # phone, context
            )))
            fout.write('\n')
