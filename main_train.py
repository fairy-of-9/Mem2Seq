import numpy as np
import logging 
import pickle

from tqdm import tqdm
from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *
from utils.utils import save_data, load_data
import yaml
import csv
from copy import deepcopy
import time
# tensorboard
from torch.utils.tensorboard import SummaryWriter 
# for clean printing
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

'''
CUDA_VISIBLE_DEVICES=2 python main_train.py -lr=0.001 -layer=3 -hdd=256 -dr=0.2 -dec=Mem2Seq -bsz=256 -ds=babi -t=6 
'''


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
# else:
#     if args['dataset']=='kvr':
#         from utils.utils_kvr import *
#         BLEU = True
#     elif args['dataset']=='babi':
#         from utils.utils_babi import *
#     else: 
#         print("You need to provide the --dataset information")

# Configure models
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0   

# load processed data
if not(os.path.exists(os.path.join(args['data'], 'processed_data.pickle'))):
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)
    processed = {
        'train': train,
        'dev': dev,
        'test': test,
        'testOOV': testOOV,
        'lang': lang,
        'max_len': max_len,
        'max_r': max_r}
    save_data(os.path.join(args['data'], 'processed_data.pickle'), processed)
else:
    print('load processed data from pickle file.')
    processed = load_data(os.path.join(args['data'], 'processed_data.pickle'))
    train, dev, test, testOOV, lang, max_len, max_r = processed['train'], processed['dev'], processed['test'], processed['testOOV'], processed['lang'], processed['max_len'], processed['max_r']

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
# init summary writer
writer = SummaryWriter(args['log'])
for epoch in range(args['epoch']):
    logging.info("Epoch:{}".format(epoch))
    print("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    total_loss = 0
    for i, data in pbar: 
        loss = model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                        len(data[1]),10.0,0.5,i==0)
        total_loss += loss
        pbar.set_description(model.print_loss())
    # add loss to the summaryWriter
    writer.add_scalar('loss', total_loss/len(train), epoch)
    # evaluation
    if((epoch+1) % int(args['evalp']) == 0):
        acc, score = model.evaluate(dev,avg_best, epoch+1, BLEU)    # 여기서 나온 score에 해당 epoch에 대한 epoch,점수들이 들어있습니다.
                                                                    # score['epoch'], score['F1'], score['dialog'], score['BLEU'], score['acc_avg']
        cp = deepcopy(score)
        scores.append(cp)
        # add values to the summaryWriter
        writer.add_scalar('f1_score', score['F1'], epoch)
        writer.add_scalar('dialog', score['dialog'], epoch)
        writer.add_scalar('bleu', score['BLEU'], epoch)
        writer.add_scalar('acc_avg', score['acc_avg'], epoch)

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

