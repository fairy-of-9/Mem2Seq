import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *
import yaml

'''
CUDA_VISIBLE_DEVICES=3 python main_test.py -dec=Mem2Seq -path=save/mem2seq-BABI/5HDD256BSZ64DR0.2L3lr0.001Mem2Seq0.9724805363321799 -bsz=32 -ds=babi -t=5
'''

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    else: 
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else: 
        print("You need to provide the --dataset information")

# Configure models
directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]

train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=int(args['batch']))

with open(os.path.join('config', 'config.yaml'), 'r') as f:
    config = yaml.load(f)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](
        config, int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)
else:
    model = globals()[args['decoder']](
        config, int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0)

acc_test = model.evaluate(test, 1e6, 0, BLEU)
print(acc_test)
if testOOV!=[]:
    acc_oov_test = model.evaluate(testOOV,1e6,0,BLEU)
    print(acc_oov_test)

