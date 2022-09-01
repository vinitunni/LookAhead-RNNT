'''
This script will recursively find all files within a folder structure and transliterate files for PER calculation.
run it as
python transliterate.py --folder <RECURSIVE FOLDER PARENT NAME>
'''

import editdistance as ed
import epitran
import os, glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Pass folder name details")
parser.add_argument('--folder', type=str, default='.', help='name of folder where the decoder should be run')

args = parser.parse_args()
# root_folder="/mnt/a99/d0/vinit/exp/espnet-0.10.4/egs/librispeech/asr1/exp_fresh"
root_folder=args.folder
ref_folds = set([ x.rstrip('ref.wrd.trn') for x in glob.glob(os.path.join(root_folder,'**','ref.wrd.trn'),recursive=True) ])
hyp_folds = set([ x.rstrip('hyp.wrd.trn') for x in glob.glob(os.path.join(root_folder,'**','hyp.wrd.trn'),recursive=True) ])
epi = epitran.Epitran('eng-Latn')
try:
    assert (ref_folds == hyp_folds)
    all_folds = ref_folds
except:
    print("Ref and hyp folders mismatch")
    all_folds = ref_folds.intersection(hyp_folds)
for folder in all_folds:
    print(folder)
    dict_all = {}
    # for ref file
    if not ((os.path.exists(os.path.join(folder,'ref.ipa.trn'))) and (os.path.exists(os.path.join(folder,'hyp.ipa.trn'))) and (os.path.exists(os.path.join(folder,'per.val')))):
        numlines = sum( 1 for line in open(os.path.join(folder,'ref.wrd.trn'),'r'))
        with open(os.path.join(folder,'ref.wrd.trn'),'r') as r:
            for line in tqdm(r,total=numlines):
                line_tmp = line.replace('(',' (')
                utt_id =line_tmp.strip().split()[-1]
                ref = ' '.join(line_tmp.strip().split()[:-1])
                dict_all[utt_id]={}
                dict_all[utt_id]['ref'] = ref
                # dict_all[utt_id]['ref_ipa'] = ' '.join(epi.trans_list(ref)).split()
                dict_all[utt_id]['ref_ipa'] = epi.transliterate(ref)
        with open(os.path.join(folder,'hyp.wrd.trn'),'r') as r:
            for line in tqdm(r,total=numlines):
                line_tmp = line.replace('(',' (')
                utt_id =line_tmp.strip().split()[-1]
                hyp = ' '.join(line_tmp.strip().split()[:-1])
                assert (utt_id in dict_all)
                dict_all[utt_id]['hyp'] = hyp
                # dict_all[utt_id]['hyp_ipa'] = ' '.join(epi.trans_list(hyp)).split()
                dict_all[utt_id]['hyp_ipa'] = epi.transliterate(hyp)

        edit_dist = 0
        num_chars = 0
        with open(os.path.join(folder,'ref.ipa.trn'),'w') as wr:
            with open(os.path.join(folder,'hyp.ipa.trn'),'w') as wh:
                for utt_id in tqdm(dict_all):
                    wr.write(dict_all[utt_id]['ref_ipa']+' '+utt_id+'\n')
                    wh.write(dict_all[utt_id]['hyp_ipa']+' '+utt_id+'\n')
                    edit_dist += ed.eval(list(dict_all[utt_id]['ref_ipa'].replace(' ','')),list(dict_all[utt_id]['hyp_ipa'].replace(' ','')))
                    num_chars += len(dict_all[utt_id]['ref_ipa'].replace(' ',''))
        per = 100.0 * edit_dist / num_chars
        with open(os.path.join(folder,'per.val'),'w') as wr:
            wr.write('PER Value is '+str(per)+'.')
        del(dict_all)
            
    else:
        print("Exists")
