
# coding: utf-8

# In[3]:


from old.fastai.text import *
import html
import pathlib
from path import Path
import numpy as np
import pandas as pd
import sklearn
#import sklearn.cross_validation
import re
import cv2
import graphviz


# In[5]:


chunksize = 12000
df_trn = pd.read_csv('data/ag_news_csv/train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv('data/ag_news_csv/test.csv', header=None, chunksize=chunksize)


# In[7]:


re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


# In[8]:


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


# In[9]:


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_
        labels += labels_
    return tok, labels


# In[11]:


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
tok_file1 = 'data/ag_news_csv/tok_trn.npy'
tok_file2 = 'data/ag_news_csv/tok_val.npy'
if (os.path.exists(tok_file1)): 
    tok_trn, trn_labels = get_all(df_trn, 1)

    tok_val, val_labels = get_all(df_val, 1)


    # In[16]:


    np.save('data/ag_news_csv/tok_trn.npy', tok_trn)
    np.save('data/ag_news_csv/tok_val.npy', tok_val)


    # In[17]:


    freq = Counter(p for o in tok_trn for p in o)
    freq.most_common(25)


    # In[18]:


    max_vocab = 60000
    min_freq = 2


    # In[19]:


    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')


    # In[21]:


    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)


    # In[22]:


    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tok_val])


    # In[23]:


    np.save('data/ag_news_csv/trn_ids.npy', trn_lm)
    np.save('data/ag_news_csv/val_ids.npy', val_lm)
    pickle.dump(itos, open('data/ag_news_csv/itos.pkl', 'wb'))


# In[5]:


trn_lm = np.load('data/ag_news_csv/trn_ids.npy')
val_lm = np.load('data/ag_news_csv/val_ids.npy')
itos = pickle.load(open('data/ag_news_csv/itos.pkl', 'rb'))


# In[6]:


vs=len(itos)
vs,len(trn_lm)


# In[7]:


em_sz,nh,nl = 400,1150,3


# In[8]:


PRE_PATH = 'wt103'
PRE_LM_PATH = 'fwd_wt103.h5'


# In[9]:


wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)


# In[10]:


enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)


# In[11]:


itos2 = pickle.load(open('itos_wt103.pkl','rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# In[12]:


new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


# In[ ]:


wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


# In[ ]:


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# In[ ]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData('data/ag_news_csv/', 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


# In[ ]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


# In[ ]:


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[ ]:


learner.model.load_state_dict(wgts)


# In[ ]:


lr=1e-3
lrs = lr


# In[ ]:


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# In[ ]:


learner.save('lm_last_ft')


# In[ ]:


learner.load('lm_last_ft')


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

learner.sched.plot()


# In[ ]:


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)


# We save the trained model weights and separately save the encoder part of the LM model as well. This will serve as our backbone in the classification task model.

# In[ ]:


learner.save('lm1')


# In[ ]:


learner.save_encoder('lm1_enc')


# In[ ]:


learner.sched.plot_loss()


# ## Classifier tokens

# The classifier model is basically a linear layer custom head on top of the LM backbone. Setting up the classifier data is similar to the LM data setup except that we cannot use the unsup movie reviews this time.

# In[ ]:


df_trn = pd.read_csv('data/ag_news_csv/train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv('data/ag_news_csv/test.csv', header=None, chunksize=chunksize)


# In[ ]:


tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)


# In[ ]:



np.save('tok_trn.npy', tok_trn)
np.save('tmp'/'tok_val.npy', tok_val)

np.save('tmp'/'trn_labels.npy', trn_labels)
np.save('tmp'/'val_labels.npy', val_labels)


# In[ ]:


tok_trn = np.load('tmp'/'tok_trn.npy')
tok_val = np.load('tmp'/'tok_val.npy')


# In[ ]:


itos = pickle.load(('tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[ ]:


trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])


# In[ ]:


np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)


# ## Classifier

# Now we can create our final model, a classifier which is really a custom linear head over our trained IMDB backbone. The steps to create the classifier model are similar to the ones for the LM.

# In[ ]:


trn_clas = np.load('tmp'/'trn_ids.npy')
val_clas = np.load('tmp'/'val_ids.npy')


# In[ ]:


trn_labels = np.squeeze(np.load('tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load('tmp'/'val_labels.npy'))


# In[ ]:


bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48


# In[ ]:


min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1


# In the classifier, unlike LM, we need to read a movie review at a time and learn to predict the it's sentiment as pos/neg. We do not deal with equal bptt size batches, so we have to pad the sequences to the same length in each batch. To create batches of similar sized movie reviews, we use a sortish sampler method invented by [@Smerity](https://twitter.com/Smerity) and [@jekbradbury](https://twitter.com/jekbradbury)
# 
# The sortishSampler cuts down the overall number of padding tokens the classifier ends up seeing.

# In[ ]:


trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)


# In[ ]:


# part 1
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])


# In[ ]:


dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5


# In[ ]:


m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[ ]:


opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# In[ ]:


learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=.25
learn.metrics = [accuracy]


# In[ ]:


lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])


# In[ ]:


lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])


# In[ ]:


wd = 1e-7
wd = 0
learn.load_encoder('lm1_enc')


# In[ ]:


learn.freeze_to(-1)


# In[ ]:


learn.lr_find(lrs/1000)
learn.sched.plot()


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[ ]:


learn.save('clas_0')


# In[ ]:


learn.load('clas_0')


# In[ ]:


learn.freeze_to(-2)


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[ ]:


learn.save('clas_1')


# In[ ]:


learn.load('clas_1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


learn.save('clas_2')


# The previous state of the art result was 94.1% accuracy (5.9% error). With bidir we get 95.4% accuracy (4.6% error).

# ## Fin

# In[ ]:


learn.sched.plot_loss()

