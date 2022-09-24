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
    import pdb; pdb.set_trace()
    folders = cer_folds



dict_all = {}

for folder in folders:
    model_name, decode_folder, _= folder.split('/')[-3:]
    if 'train' not in model_name:
        import pdb; pdb.set_trace()
    if model_name not in dict_all:
        dict_all[model_name] = {}
    if decode_folder not in dict_all[model_name]:
        dict_all[model_name][decode_folder]={}
    # For PER
    file_name = 'per.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]['per']=(open(os.path.join(folder,file_name),'r').readline().split(' ')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]['per']='NA'
    # For CER
    file_name = 'cer.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]['cer']=(open(os.path.join(folder,file_name),'r').readline().split('>')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]['cer']='NA'
    # For WER
    file_name = 'wer.val'
    if os.path.exists(os.path.join(folder,file_name)):
        dict_all[model_name][decode_folder]['wer']=(open(os.path.join(folder,file_name),'r').readline().split('>')[-1]).rstrip('.')
    else:
        dict_all[model_name][decode_folder]['wer']='NA'

with open(os.path.join(root_folder,'xer_values.json'),'w') as w:
    json.dump(dict_all,w)
import pdb; pdb.set_trace()
print("Dummy")

with open(os.path.join(root_folder,'xer_values_mcv.csv'),'w') as w:
    w.write("Model Name, per, cer, wer, Path\n")
    for model in dict_all:
        for decode_folder in dict_all[model]:
            if "mcv" in decode_folder:
                w.write(model+','+dict_all[model][decode_folder]['per']+','+dict_all[model][decode_folder]['cer']+','+dict_all[model][decode_folder]['wer']+','+decode_folder+'\n')
with open(os.path.join(root_folder,'xer_values_libtest.csv'),'w') as w:
    w.write("Model Name, per, cer, wer, Path\n")
    for model in dict_all:
        for decode_folder in dict_all[model]:
            if "test_clean" in decode_folder:
                w.write(model+','+dict_all[model][decode_folder]['per']+','+dict_all[model][decode_folder]['cer']+','+dict_all[model][decode_folder]['wer']+','+decode_folder+'\n')
with open(os.path.join(root_folder,'xer_values_libdev.csv'),'w') as w:
    w.write("Model Name, per, cer, wer, Path\n")
    for model in dict_all:
        for decode_folder in dict_all[model]:
            if "dev_clean" in decode_folder:
                w.write(model+','+dict_all[model][decode_folder]['per']+','+dict_all[model][decode_folder]['cer']+','+dict_all[model][decode_folder]['wer']+','+decode_folder+'\n')
