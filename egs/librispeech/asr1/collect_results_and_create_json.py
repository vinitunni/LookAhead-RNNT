'''\
This script will recursively traverse results directory and store results as json

@vinit
'''
import os, glob, json
root_folder=os.getcwd()
cer_folds = set([ x.rstrip('cer.val') for x in glob.glob(os.path.join(root_folder,'**','**','cer.val')) ])
per_folds = set([ x.rstrip('per.val') for x in glob.glob(os.path.join(root_folder,'**','**','per.val')) ])
wer_folds = set([ x.rstrip('wer.val') for x in glob.glob(os.path.join(root_folder,'**','**','wer.val')) ])

try:
    assert (cer_folds == per_folds and per_folds == wer_folds)
    folders = cer_folds
except:
    folders = cer_folds.intersection(per_folds.intersection(wer_folds))
    print("there is folder discrepancy. Analyze??")



dict_all = {}

for folder in folders:
    model_name, decode_folder, _= folder.split('/')[-3:]
    if 'train' not in model_name:
        import pdb; pdb.set_trace()
    if model_name not in dict_all:
        dict_all[model_name] = {}
        # For PER
    file_name = 'per.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]=(open(os.path.join(folder,file_name),'r').readline().split('>')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]='NA'
    # For CER
    file_name = 'cer.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]=(open(os.path.join(folder,file_name),'r').readline().split('>')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]='NA'
    # For WER
    file_name = 'wer.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]=(open(os.path.join(folder,file_name),'r').readline().split('>')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]='NA'

with open(os.path.join(root_folder,'xer_values.json'),'w') as w:
    json.dump(dict_all,w)
import pdb; pdb.set_trace()
print("Dummy")
