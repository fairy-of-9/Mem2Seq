import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *
import yaml
import csv
from copy import deepcopy

'''
CUDA_VISIBLE_DEVICES=2 python main_train.py -lr=0.001 -layer=3 -hdd=256 -dr=0.2 -dec=Mem2Seq -bsz=256 -ds=babi -t=6 
'''

import time

now = time.localtime()
cur_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
print("START")
print(cur_time)

logging.basicConfig(filename='./log/{}.log'.format(cur_time))



BLEU = False

with open(os.path.join('config', 'config.yaml'), 'r') as f:
    config = yaml.load(f)

print(config)

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
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0   
### LOAD DATA
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](config, int(args['hidden']),
                                        max_len,max_r,lang,args['path'],args['task'],
                                        lr=float(args['learn']),
                                        n_layers=int(args['layer']), 
                                        dropout=float(args['drop']),
                                        unk_mask=bool(int(args['unk_mask']))
                                    )
else:
    model = globals()[args['decoder']](config, int(args['hidden']),
                                    max_len,max_r,lang,args['path'],args['task'],
                                    lr=float(args['learn']),
                                    n_layers=int(args['layer']), 
                                    dropout=float(args['drop'])
                                )


scores = []
args['evalp'] = 5
n_epoch = 500

for epoch in range(n_epoch):
    logging.info("Epoch:{}".format(epoch))
    print("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar: 
        model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                        len(data[1]),10.0,0.5,i==0)
        pbar.set_description(model.print_loss())

    if((epoch+1) % int(args['evalp']) == 0):
        acc, score = model.evaluate(dev,avg_best, epoch+1, BLEU)    # 여기서 나온 score에 해당 epoch에 대한 epoch,점수들이 들어있습니다.
                                                                    # score['epoch'], score['F1'], score['dialog'], score['BLEU'], score['acc_avg']
        cp = deepcopy(score)
        scores.append(cp)


        print("cur_BEST")

        if 'Mem2Seq' in args['decoder']:
            model.scheduler.step(acc)
        if(acc >= avg_best):
            print('better acc {} -> {}'.format(avg_best, acc))
            avg_best = acc
            cnt=0
        else:
            cnt+=1
        if(cnt == 5): break
        if(acc == 1.0): break 



now = time.localtime()
cur_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
print("END")
print(cur_time)


with open('{}.csv'.format(cur_time), 'w', newline='', encoding='UTF-8') as f:
    makewrite = csv.writer(f)
    makewrite.writerow(['epoch', 'dialog', 'f1', 'BLEU', 'acc_avg'])
    for i, score in enumerate(scores):
        row = [(i+1)*int(args['evalp']), score['dialog'], score['F1'], score['BLEU'], score['acc_avg']]
        makewrite.writerow(row)

