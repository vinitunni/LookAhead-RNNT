#!/usr/bin/env python3
import editdistance as ed
import os, glob
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")

root_folder=os.getcwd()
folders = set([ x.rstrip('result.wrd.txt') for x in glob.glob(os.path.join(root_folder,'**','result.wrd.txt'),recursive=True) ])
for folder in folders:
    try:
        print(folder)
        if not((os.path.exists(os.path.join(folder,'cer.val'))) and (os.path.exists(os.path.join(folder,'wer.val'))) and (os.path.exists(os.path.join(folder,'ent.cer.val'))) and (os.path.exists(os.path.join(folder,'ent.wer.val')))):
            filename = os.path.join(folder,'result.wrd.txt')
            ref=[]
            hyp=[]
            utt_id=[]

            for w in open(filename, 'r').readlines():
                if ('REF:' in w):
                    ref.append(' '.join(w.replace('*','').split()[1:]).lower())
                elif("HYP:" in w):
                    hyp.append(' '.join(w.replace('*','').split()[1:]).lower())
                elif("id: " in w):
                    utt_id.append(w.split()[1])
                
                


            assert(len(ref) == len(hyp))

            ed_char=[]
            len_char=[]

            ed_wrd=[]
            len_wrd=[]

            ed_entity_wrd=[]
            len_entity_wrd=[]
            ed_entity_char=[]
            len_entity_char=[]

            with open(os.path.join(folder,'ref.ent.trn'),'w') as wr:
                with open(os.path.join(folder,'hyp.ent.trn'),'w') as wh:
                    for i in tqdm(range(len(ref))):
                        ed_char.append(ed.eval(''.join(ref[i].split()),''.join(hyp[i].split())))
                        ed_wrd.append(ed.eval(ref[i].split(),hyp[i].split()))

                        len_char.append(len(''.join(ref[i].split())))
                        len_wrd.append(len(ref[i].split()))
                        # import pudb; pu.db
                        doc = nlp(ref[i])
                        if len(doc.ents)>0:
                            wr.write(ref[i]+' '+utt_id[i]+'\n')
                            wh.write(hyp[i]+' '+utt_id[i]+'\n')
                            ed_entity_char.append(ed.eval(''.join(ref[i].split()),''.join(hyp[i].split())))
                            ed_entity_wrd.append(ed.eval(ref[i].split(),hyp[i].split()))

                            len_entity_char.append(len(''.join(ref[i].split())))
                            len_entity_wrd.append(len(ref[i].split()))
                            



            with open(os.path.join(folder,'cer.val'),'w') as w:
                w.write('CER is --->'+ str(100.0*sum(ed_char)/sum(len_char)))
                print('CER is --->', 100.0*sum(ed_char)/sum(len_char))
            with open(os.path.join(folder,'wer.val'),'w') as w:
                w.write('WER is --->'+ str(100.0*sum(ed_wrd)/sum(len_wrd)))
                print('WER is --->', 100.0*sum(ed_wrd)/sum(len_wrd))
            with open(os.path.join(folder,'ent.cer.val'),'w') as w:
                w.write('Entity CER is --->'+ str(100.0*sum(ed_entity_char)/sum(len_entity_char)))
                print('Entity CER is --->', 100.0*sum(ed_entity_char)/sum(len_entity_char))
            with open(os.path.join(folder,'ent.wer.val'),'w') as w:
                w.write('Entity WER is --->'+ str(100.0*sum(ed_entity_wrd)/sum(len_entity_wrd)))
                print('Entity WER is --->', 100.0*sum(ed_entity_wrd)/sum(len_entity_wrd))
        else:
            print("Exists!!")
    except Exception as e:
        print('Some error happened !! \n')
        print(folder)
        print(e)
