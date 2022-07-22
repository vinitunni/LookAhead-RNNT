#!/bin/python
import editdistance as ed
import os, glob
from tqdm import tqdm

root_folder=os.getcwd()
folders = set([ x.rstrip('result.wrd.txt') for x in glob.glob(os.path.join(root_folder,'**','result.wrd.txt')) ])
for folder in folders:
    print(folder)
    filename = os.path.join(folder,'result.wrd.txt')
    ref=[]
    hyp=[]

    for w in open(filename, 'r').readlines():
        if ('REF:' in w):
            ref.append(' '.join(w.replace('*','').split()[1:]).lower())
        elif("HYP:" in w):
            hyp.append(' '.join(w.replace('*','').split()[1:]).lower())


    assert(len(ref) == len(hyp))

    ed_char=[]
    len_char=[]

    ed_wrd=[]
    len_wrd=[]

    for i in range(len(ref)):
        ed_char.append(ed.eval(''.join(ref[i].split()),''.join(hyp[i].split())))
        ed_wrd.append(ed.eval(ref[i].split(),hyp[i].split()))

        len_char.append(len(''.join(ref[i].split())))
        len_wrd.append(len(ref[i].split()))
        # import pudb; pu.db



    with open(os.path.join(folder,'cer.val'),'w') as w:
        w.write('CER is --->'+ str(100.0*sum(ed_char)/sum(len_char)))
        print('CER is --->', 100.0*sum(ed_char)/sum(len_char))
    with open(os.path.join(folder,'wer.val'),'w') as w:
        w.write('WER is --->'+ str(100.0*sum(ed_wrd)/sum(len_wrd)))
        print('WER is --->', 100.0*sum(ed_wrd)/sum(len_wrd))
