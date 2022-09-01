'''
This script gives us the panphon features weighted edit distance between two sentences thus being a better metric
Running the script from scratch foreach folder
@vinit
'''

import epitran, panphon.distance
import editdistance as ed
import  os, glob
from tqdm  import tqdm
import pickle
dst = panphon.distance.Distance()
epi=epitran.Epitran('eng-Latn')

ref_folders=[x.rstrip('ref.wrd.trn') for x in glob.glob(os.path.join(os.getcwd(),'**','ref.wrd.trn'),recursive=True)]
hyp_folders=[x.rstrip('hyp.wrd.trn') for x in glob.glob(os.path.join(os.getcwd(),'**','hyp.wrd.trn'),recursive=True)]

folders=set(ref_folders).intersection(hyp_folders)

for folder in folders:
	# Check of both ref and hyp of IPA are there (We might require word level IPAS btw, so fix that)
	print(folder)
	if not os.path.exists(os.path.join(folder,'panphon.pkl')):
		dict_utts={}
		counter=0
		with open(os.path.join(folder,'ref.wrd.trn'),'r') as r:
			for line in r:
				utt_id= line.split('(')[-1].strip('()')
				assert utt_id not in dict_utts
				dict_utts[utt_id]={}
				dict_utts[utt_id]['ref']=str(line.split('(')[0])
				counter+=1
		with open(os.path.join(folder,'hyp.wrd.trn'),'r') as r:
			for line in r:
				utt_id= line.split('(')[-1].strip('()')
				dict_utts[utt_id]['hyp']=str(line.split('(')[0])
				counter+=1
		for utt_id ,_ in tqdm(dict_utts.items()):
			dict_utts[utt_id]['wrd_len']=len(dict_utts[utt_id]['ref'].split())
			dict_utts[utt_id]['chr_len']=len(dict_utts[utt_id]['ref'].replace(' ',''))
			dict_utts[utt_id]['ref_ipa']=epi.transliterate(dict_utts[utt_id]['ref'].strip())
			dict_utts[utt_id]['hyp_ipa']=epi.transliterate(dict_utts[utt_id]['hyp'].strip())
			# dict_utts[utt_id]['weighted-per']=dst.weighted_feature_edit_distance(epi.transliterate(dict_utts[utt_id]['ref']),epi.transliterate(dict_utts[utt_id]['hyp']))
			dict_utts[utt_id]['weighted-per']=dst.weighted_feature_edit_distance(dict_utts[utt_id]['ref_ipa'],dict_utts[utt_id]['hyp_ipa'])

		print("Dummy")
		with open(os.path.join(folder,'panphon.pkl'),'wb') as w:
		    pickle.dump(dict_utts,w)
	else:
		print("Exists")

	# If IPA present, then confirm it is word wise and then u can use the weighted distance straight away
