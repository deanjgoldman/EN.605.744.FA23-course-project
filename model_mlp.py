import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import time
import json
import os

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import torch
from arch import MLP

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics


#############
# Constants #
#############
data_dir = os.path.abspath('data')


#############
# Utilities #
#############
def get_dataframe(fp):
    fp = os.path.join(fp)
    names = [
        'assessment',
        'docid',
        'title',
        'authors',
        'journal',
        'issn',
        'year',
        'language',
        'abstract',
        'keywords'
    ]
    df = pd.read_csv(fp, sep='\t', names=names)
    return df

def display_stats(y_true, y_pred, dir_output, title):
    stats = metrics.precision_recall_fscore_support(y_true, y_pred)
    support = np.unique(y_true, return_counts=True)[1]
    fmt = '{:<10} {:<10} {:<10}'
    header = fmt.format('Metric', '[-1]', '[1]')
    print('-'*len(header))
    print(title)
    print('-'*len(header))
    print(header)
    print('-'*len(header))
    print(fmt.format('Precision', np.round(stats[0][0], 4), np.round(stats[0][1], 4)))
    print(fmt.format('Recall', np.round(stats[1][0], 4), np.round(stats[1][1], 4)))
    print(fmt.format('F1-score', np.round(stats[2][0], 4), np.round(stats[2][1], 4)))
    print(fmt.format('Support', int(support[0]), int(support[1])))
    print()
    with open(os.path.join(dir_output, f'stats-{title}.txt'), 'w') as out:
        for stat in stats:
            out.write(str(np.round(stat[0], 4))+','+str(np.round(stat[1], 4))+'\n')

def plot_performance(y_true, y_prob, dir_output, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    y_true = (y_true > 0).astype(int)
    x,y,_ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(x, y)
    ax[0].grid()
    ax[0].plot(x, y, label = 'auc: {:.4f}'.format(auc))
    ax[0].legend()
    ax[0].set_title('ROC Curve')
    ax[0].set_ylabel('TPR')
    ax[0].set_xlabel('FPR')

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    ap = metrics.average_precision_score(y_true, y_prob, pos_label=1)
    auc = metrics.auc(recall, precision)
    ax[1].grid()
    ax[1].plot(recall, precision, color='purple', label = 'ap: {:.4f}\nauc: {: .4f}'.format(ap, auc))
    ax[1].legend()
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')

    fig.suptitle(f'plot performance {title}')
    plt.savefig(os.path.join(dir_output, f'plot-performance-{title}.png'))
    plt.close('all')
    return

def dataloader(df, vocabulary=set(), ns=[1,], tf_thresh=15, add_authors=True):

    def make_ngrams(text, ns=[1, 2]):
        ngrams = []
        elements = text.split(' ')
        elements = [el for el in elements if not el in ('', ' ')]
        for n in ns:
            for i in range(len(elements)-n):
                ngram = '$'.join(elements[i:i+n])
                ngrams.append(ngram)
        return ' '.join(ngrams)

    def _clean(token):
        for char in '()/[]:.,?!<>;':
            token = token.replace(char, ' ')
        for symbol in ['--', '\\t']:
            token = token.replace(symbol, ' ')
        return token.strip().lower()

    def _tokenize(strings):
        for string in strings:
            if string.isdigit():
                continue
            if len(string) <= 2:
                continue
            if string in stopwords.words('english'):
                continue
            if all([
                (s in stopwords.words('english')) or (s in ('', ' '))\
                 for s in string.split('$')]):
                continue
            if (sum([char.isdigit() for char in string]) / len(string)) >= 0.55:
                continue
            yield string

    def _compile_authors(authors):
        raw_split = authors.split(';')
        return ' '.join([_clean(author).replace(' ', '') for author in raw_split])

    if add_authors:
        df['text'] = df.apply(lambda row: str(row['title'])+' '+str(row['abstract'])+' '+str(row['keywords'])+' '+str(_compile_authors(str(row['authors']))), axis=1)
    else:
        df['text'] = df.apply(lambda row: str(row['title'])+' '+str(row['abstract'])+' '+str(row['keywords']), axis=1)#+' '+str(_compile_authors(str(row['authors']))), axis=1)

    df['cleaned'] = df['text'].apply(lambda text: _clean(text))
    df['ngrams'] = df['cleaned'].apply(lambda text: make_ngrams(text, ns=ns).split(' '))
    df['tokens'] = df['ngrams'].apply(lambda text: list(_tokenize(text)))
    # import pdb; pdb.set_trace();
    # create a vocabulary if none is provided
    if len(vocabulary) == 0:
        return_vocabulary = True
        tokens_count = {}
        for i, token_set in enumerate(df['tokens']):
            for token in token_set:
                if not tokens_count.get(token):
                    tokens_count[token] = 0
                tokens_count[token] += 1
        vocabulary = [token for token, count in tokens_count.items() if count >= tf_thresh]
        vocabulary.append('<pad>') # ?
        print(f'Number of tokens: {len(vocabulary)}')
    else:
        return_vocabulary = False
    
    # print(f'Populating feature vectors...')

    x_train = np.zeros((len(df), len(vocabulary)))
    for i, token_seq in enumerate(df['tokens']):
        for j, token in enumerate(token_seq):
            if token in vocabulary:
                x_train[i, vocabulary.index(token)] = 1
            else:
                pass
    # import pdb; pdb.set_trace()
    y_train = df['assessment'].to_numpy()

    if return_vocabulary:
        return x_train, y_train, vocabulary
    else:
        return x_train, y_train


def train(net, x, y, training_params):
    device = 'cuda:0'
    epochs = training_params['epochs']
    bsz = training_params['batch_size']
    lr = training_params['learning_rate']
    mod = training_params['lr_decay_freq']
    lr_decay = training_params['lr_decay']
    fmt = '{:<5} {:<8} {:<8}'
    print(fmt.format('epoch', 'cls-loss', 'test-acc'))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    weight = torch.Tensor(training_params['weight']).to(torch.device(device))
    loss_func = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')

    batch_iters = x.shape[0] // bsz
    

    n = x.shape[0]
    n_test = int(n * 0.2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_test)
    
    del x
    del y

    for e in range(epochs):

        net.to(torch.device(device))
        net.train()

        avg_cost = 0.0
        if (e > 0) and ((e % mod) == 0):
            for g in optimizer.param_groups:
                g['lr'] *= lr_decay

        for i in range(batch_iters):
            indices = np.random.choice(np.arange(x_train.shape[0]), bsz)

            xi = torch.Tensor(x_train[indices]).to(torch.device(device)).to(torch.long)
            yi = torch.Tensor(y_train[indices] > 0).to(torch.device(device)).to(torch.long)

            optimizer.zero_grad()

            yhat = net(xi)
            cost = loss_func(yhat, yi)

            cost.backward()
            optimizer.step()

            avg_cost += cost.detach().item() / batch_iters

        pred, prob = predict(net, x_test)
        acc = sum(pred == y_test) / y_test.shape[0] 
        fmt = '{:<5} {:<8.5f} {:<8.5f}'
        print(fmt.format(e, avg_cost, acc))

    return net

def predict(net, x):

    x = torch.Tensor(x).to('cpu').to(torch.long)

    net.to(torch.device('cpu'))
    net.eval()
    
    pred = torch.zeros(x.shape[0], 1)
    prob = torch.zeros(x.shape[0], 2)

    with torch.no_grad():
        for i in range(x.shape[0]-1):
            yhat = torch.softmax(net(x[i:i+1]), 1)
            pred[i] = torch.argmax(yhat, 1)
            prob[i] = yhat

    pred = pred.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()
    # convert to {-1, 1}
    pred = 2 * (pred - 0.5)
    return pred[:, 0], prob



def main(params):

    upsample_size = params['upsample_size']
    upsample_iters = params['upsample_iters']
    ngrams = params['ngrams']
    tf_thresh = params['tf_thresh']
    experiment = f'model-mlp-boost-init-h-{params["dim_h"]}-ngrams-{"-".join([str(x) for x in ngrams])}-w-{int(params["weight"][1])}'
    
    print(experiment+'\n')

    dir_output = os.path.join(os.path.abspath('output'), experiment)
    
    # skip if experimental results exists
    # if os.path.exists(dir_output):
    #     if os.path.exists(os.path.join(dir_output, 'pred_frame.csv')):
    #         return

    os.makedirs(dir_output, exist_ok=True)

    ############
    # Training #
    ############
    train_fp = os.path.join(data_dir, 'phase1.train.shuf.tsv')
    df = get_dataframe(train_fp)

    # load up dev set for inference later
    dev_fp = os.path.join(data_dir, 'phase1.dev.shuf.tsv')
    dev_df = get_dataframe(dev_fp)

    nfolds = 5
    fold_idx = np.random.choice(np.arange(nfolds), len(df))
    df['fold'] = fold_idx
    for fold in range(nfolds):

        train_df = df[fold_idx != fold]
        train_df = train_df.sample(int(len(train_df) * 0.8))
        
        # del df
        # upsample underrepresented class
        to_upsample = train_df[train_df['assessment'] == 1]
        for i in range(upsample_iters):
            train_df = pd.concat((train_df, to_upsample.sample(upsample_size)))
     
        x_train, y_train, vocabulary = dataloader(
            train_df,
            ns=ngrams,
            tf_thresh=tf_thresh)

        # import pdb; pdb.set_trace();
        net = MLP(
            dim_in=len(vocabulary),
            dim_h=params['dim_h'],
            dim_out=2)
        
        del train_df
        net = train(net, x_train, y_train, params)

        # stash network weights
        torch.save(net.state_dict(), os.path.join(dir_output, f'state_dict-f{fold}.pt'))
        # stash vocabulary for this run
        with open(os.path.join(dir_output, f'vocabulary-f{fold}.csv'), 'w') as out:
            for term in vocabulary:
                out.write(term+'\n')

        # y_true = y_train
        # y_pred, y_prob = predict(net, x_train)
        # display_stats(y_true, y_pred, dir_output=dir_output, title='train')

        del x_train
        del y_train
        # del y_pred
        # del y_prob

        ########################
        # Cross-fold inference #
        ########################
        df_out_of_fold = df[fold_idx == fold]
        x_test, y_test = dataloader(
            df_out_of_fold,
            vocabulary=vocabulary,
            ns=ngrams,
            tf_thresh=tf_thresh)

        y_pred, y_prob = predict(net, x_test)

        display_stats(y_test, y_pred, dir_output=dir_output, title=f'fold-{fold}')
        plot_performance(y_test, y_prob[:, 1], dir_output=dir_output, title=f'fold-{fold}')

        # import pdb; pdb.set_trace()
        df.loc[fold_idx == fold, 'yz'] = y_prob[:, 1]

        del x_test
        del y_test
        del y_pred
        del y_prob

        ###########
        # Dev Set #
        ###########
        x_dev, y_dev = dataloader(
            dev_df,
            vocabulary=vocabulary,
            ns=ngrams,
            tf_thresh=tf_thresh)

        y_pred, y_prob = predict(net, x_dev)
        dev_df[f'score-f{fold}'] = y_prob[:, 1]

        del y_pred
        del y_prob
        del x_dev
        del net

    # import pdb; pdb.set_trace();

    y_prob_ens = np.mean([dev_df[f'score-f{fold}'] for fold in range(nfolds)], 0)
    y_pred_ens = 2 * (np.round(y_prob_ens) - 0.5)
    display_stats(y_dev, y_pred_ens, title='dev', dir_output=dir_output)
    plot_performance(y_dev, y_prob_ens, dir_output=dir_output, title='dev')

    pred_frame = dev_df[['docid', 'title', 'abstract', 'keywords', 'assessment', 'authors'] + [f'score-f{fold}' for fold in range(nfolds)]]
    filename = os.path.join(dir_output, 'pred_frame.csv')
    pred_frame.to_csv(filename, index=False)

    oof_frame = df[['docid', 'yz', 'fold']]
    filename = os.path.join(dir_output, 'oof_frame.csv')
    oof_frame.to_csv(filename, index=False)

    filename = os.path.join(dir_output, 'params.json')
    with open(filename, 'w') as out:
        json.dump(params, out)

if __name__ == '__main__':

    ngram_sets = ([1,],)
    dim_hs = (32,)

    param_space = [
        ngram_sets,
        dim_hs,
    ]

    for (ngrams, dim_h) in itertools.product(*param_space):
 
        params = dict(
            dim_h=dim_h,
            epochs=20,
            batch_size=64,
            learning_rate=1e-3,
            lr_decay_freq=10,
            lr_decay=0.5,
            upsample_size=80,
            upsample_iters=50,
            ngrams=ngrams,
            weight=[1e-0, 1e+2],
            tf_thresh=15)
        main(params)