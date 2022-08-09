
import editdistance as ed
import os, glob
from tqdm import tqdm

# Getting high error values for RNNT pretrained
dict_utts_rnnt={}
folder_name="/mnt/a99/d0/vinit/exp/espnet-0.10.4/egs/librispeech/asr1/exp_pretrained/train_clean_960_sp_pytorch_train_conformer-rnn_transducer_specaug_300bpe_fromLibri100SOTA_checkNaselineTimingOnly/decode_mcv_en_valid_indAccent_trunc_model.last10.avg.best_decode_alsd_nolm"
tmp_file_name="hyp.ent.trn"
file_name=os.path.join(folder_name,tmp_file_name)
with open (file_name, 'r') as r:
    for line in r:
        utt_id = line.split()[-1]
        if utt_id not in dict_utts_rnnt:
            dict_utts_rnnt[utt_id] = {}
        dict_utts_rnnt[utt_id]['hyp']=' '.join(line.split()[:-1])

        

tmp_file_name="ref.ent.trn"
file_name=os.path.join(folder_name,tmp_file_name)
with open (file_name, 'r') as r:
    for line in r:
        utt_id = line.split()[-1]
        if utt_id not in dict_utts_rnnt:
            dict_utts_rnnt[utt_id] = {}
        dict_utts_rnnt[utt_id]['ref']=' '.join(line.split()[:-1])

        
for utts in dict_utts_rnnt:
    ref=dict_utts_rnnt[utts]['ref']
    hyp=dict_utts_rnnt[utts]['hyp']
    dict_utts_rnnt[utts]['wer']= 1.0 * ed.eval(ref.split(),hyp.split())/len(ref.split())
    dict_utts_rnnt[utts]['cer']= 1.0 * ed.eval(ref.replace(' ',''),hyp.replace(' ',''))/len(ref.replace(' ',''))
    if 'wer' not in dict_utts_rnnt[utts]:
        print("Dummy")

counter=0
with open("temp_wer_ent_vals",'w') as w:
    for utts in dict_utts_rnnt:
        if 'wer' not in dict_utts_rnnt[utts]:
            counter+=1
            print("Dummy")
        else:
            w.write(utts+' '+str(dict_utts_rnnt[utts]['wer'])+'\n')

sorted_utts_rnnt=sorted(dict_utts_rnnt.items(),key=lambda x:x[1]['wer'],reverse=True)

# Getting high error values for conformer-CTC pretrained
dict_utts_conformer={}
folder_name="/mnt/a99/d0/vinit/exp/espnet-0.10.4/egs/librispeech/asr1/exp_pretrained/train_clean_960_sp_pytorch_train_pytorch_conformer_large_specaug_300bpe_fromLibri100SOTA_checkNaselineTimingOnly/decode_mcv_en_valid_indAccent_trunc_model.val10.avg.best_decode_alsd_nolm"
tmp_file_name="hyp.ent.trn"
file_name=os.path.join(folder_name,tmp_file_name)
with open (file_name, 'r') as r:
    for line in r:
        utt_id = line.split()[-1]
        if utt_id not in dict_utts_conformer:
            dict_utts_conformer[utt_id] = {}
        dict_utts_conformer[utt_id]['hyp']=' '.join(line.split()[:-1])

        

tmp_file_name="ref.ent.trn"
file_name=os.path.join(folder_name,tmp_file_name)
with open (file_name, 'r') as r:
    for line in r:
        utt_id = line.split()[-1]
        if utt_id not in dict_utts_conformer:
            dict_utts_conformer[utt_id] = {}
        dict_utts_conformer[utt_id]['ref']=' '.join(line.split()[:-1])

        
for utts in dict_utts_conformer:
    ref=dict_utts_conformer[utts]['ref']
    hyp=dict_utts_conformer[utts]['hyp']
    dict_utts_conformer[utts]['wer']= 1.0*ed.eval(ref.split(),hyp.split())/len(ref.split())
    dict_utts_conformer[utts]['cer']= 1.0 * ed.eval(ref.replace(' ',''),hyp.replace(' ',''))/len(ref.replace(' ',''))
    if 'wer' not in dict_utts_conformer[utts]:
        print("Dummy")

counter=0
with open("temp_wer_ent_vals",'w') as w:
    for utts in dict_utts_conformer:
        if 'wer' not in dict_utts_conformer[utts]:
            counter+=1
            print("Dummy")
        else:
            w.write(utts+' '+str(dict_utts_conformer[utts]['wer'])+'\n')

sorted_utts_conformer=sorted(dict_utts_conformer.items(),key=lambda x:x[1]['wer'],reverse=True)
folder_name="/mnt/a99/d0/vinit/exp/espnet-0.10.4/egs/librispeech/asr1/exp_pretrained/"
tmp_file_name="report_pretrained_conformer_vs_transder_indAccent_normErrors.txt"
file_name=os.path.join(folder_name,tmp_file_name)
with open(file_name,'w') as w:
    w.write("Refs and hyps in the order where transducer error is higher and entities are present in the model. First hyp is rnnt second is conformer \n")
    for utt_id, value in sorted_utts_rnnt:
        w.write("utt_id: "+utt_id+'\n')
        w.write("REF: "+value['ref']+'\n')
        w.write("HYP1: "+value['hyp']+'\n')
        w.write("HYP2: "+dict_utts_conformer[utt_id]['hyp']+'\n')
        w.write('WER1: '+str(value['wer'])+'. WER2: '+str(dict_utts_conformer[utt_id]['wer']) +'CER1: '+str(value['cer'])+'. CER2: '+str(dict_utts_conformer[utt_id]['cer']) +'\n\n')
    w.write("#################################################\n")
    w.write("#################################################\n")
    w.write("#################################################\n")
    w.write("Refs and hyps in the order where conformer error is higher and entities are present in the model. First hyp is conformer second is rnnt \n")
    for utt_id, value in sorted_utts_conformer:
        w.write("utt_id: "+utt_id+'\n')
        w.write("REF: "+value['ref']+'\n')
        w.write("HYP1: "+value['hyp']+'\n')
        w.write("HYP2: "+dict_utts_rnnt[utt_id]['hyp']+'\n\n\n')
        w.write('WER1: '+str(value['wer'])+'. WER2: '+str(dict_utts_rnnt[utt_id]['wer']) +'CER1: '+str(value['cer'])+'. CER2: '+str(dict_utts_rnnt[utt_id]['cer']) +'\n\n')
        
import pdb; pdb.set_trace()
print("Dummy")
