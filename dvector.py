
__author__ = "Abbas Khosravani"
__email__ = "abbas.khosravani@gmail.com"
__version__ = "1.0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras import metrics, losses
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
# from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from pyannote.metrics.binary_classification import det_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kaldi_io import *
from model_new import *
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
from utils import cep_sliding_norm
import struct
import h5py
from random import shuffle

from scipy.special import erfcinv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.nan)

batch_nspks = 20
batch_spk_samples = 10
batch_size = batch_nspks * batch_spk_samples


def train_model(niter=200):
    model, generator = CNN_Kaldi(
        input_dim=23,
        batch_nspks=batch_nspks,
        batch_spk_samples=batch_spk_samples)
    model.load_weights('./models/model_cnn_kaldi_175.h5', by_name=True, skip_mismatch=True)
    # generator.save('./models/generator_cnn_kaldi_175.h5')

    opt = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5.)
    # opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0)
    # opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=5.)
    model.compile(
        optimizer=opt,
        loss={'triplet_loss': None},
        loss_weights={'triplet_loss': 1.0})

    for iter in range(176, niter + 1):
        print("Iteration: {0}/{1}".format(iter, niter))
        spkidxs = spk_sort(
                generator,
                st=0, ed=4800,
                batch_nspks=batch_nspks,
                batch_spk_samples=batch_spk_samples,
                timestep=2000,
                seed=iter)        

        model.fit_generator(
            kaldi_data_generator(
                spkidxs,
                st=0, ed=4800,
                batch_nspks=batch_nspks,
                batch_spk_samples=batch_spk_samples,
                timestep=2000,
                seed=iter),
            steps_per_epoch=4800,
            epochs=1,
            max_queue_size=50,
            validation_data=kaldi_data_generator(
                st=4800, ed=5100,
                batch_nspks=batch_nspks,
                batch_spk_samples=batch_spk_samples,
                timestep=2000,
                seed=iter + 1),
            validation_steps=64)

        model.save_weights('./models/model_cnn_kaldi_' + '{0:03d}'.format(iter) + '.h5')
        generator.save('./models/generator_cnn_kaldi_' + '{0:03d}'.format(iter) + '.h5')
        print('model saved in ./models/model_cnn_kaldi_' + '{0:03d}'.format(iter) + '.h5')

        evaluate_XY(generator, st=4800, ed=5100, timestep=1000)

        winlen = 1000
        path = '../sre16/v2/data/sre16_eval_test/'
        dvector_path = '../sre16/v2/exp/dvectors_sre16_eval_test/dvector_5s.ark'
        count = 0
        with open(dvector_path,'w') as f:
            for utt in kaldi_data(path=path):
                if len(utt[0]) < 1.5 * winlen:
                    X = generator.predict_on_batch(utt[0][np.newaxis, ...])
                else:
                    X = np.asarray([utt[0][x:x + winlen] for x in range(0, len(utt[0]) - winlen + 1, winlen / 2)])
                    X = generator.predict_on_batch(X)
                write_vec_flt(f, np.mean(X, axis=0), key=utt[1])
                count += 1
        
        path = '../sre16/v2/data/sre16_eval_enroll/'
        dvector_path = '../sre16/v2/exp/dvectors_sre16_eval_enroll/dvector_5s.ark'
        count = 0
        with open(dvector_path,'w') as f:
            for utt in kaldi_data(path=path):
                if len(utt[0]) < 1.5 * winlen:
                    X = generator.predict_on_batch(utt[0][np.newaxis, ...])
                else:
                    X = np.asarray([utt[0][x:x + winlen] for x in range(0, len(utt[0]) - winlen + 1, winlen / 2)])
                    X = generator.predict_on_batch(X)
                write_vec_flt(f, np.mean(X, axis=0), key=utt[1])
                count += 1
        

def data_generator(timestep=500, st=0, ed=1000, batch_nspks=None, batch_spk_samples=None, seed=0):
    batch_size = batch_nspks * batch_spk_samples
    nthreads = 16
    nspks = 5872
    steps = int(np.ceil(nspks / (nthreads + 0.)))
    datasets, dataset_idx, tspks = [], [], []
    for i in range(nthreads):
        dataPath = "../dataset_train_" + \
            str(i * steps) + "-" + str(steps * i + steps) + ".h5"
        dataset = h5py.File(dataPath, "r")
        spks = dataset['train'].keys()
        tspks.extend(spks)
        datasets.append(dataset)
        dataset_idx.extend([i] * len(spks))

    tspks = tspks[st:ed]
    dataset_idx = dataset_idx[st:ed]

    nspks = len(tspks)
    np.random.seed(seed)

    while True:
        feat, label, idx = [], [], 0
        gidx = np.random.permutation(nspks)
        for spk_idx in gidx:
            spk = datasets[dataset_idx[spk_idx]]['train'][tspks[spk_idx]]
            utts = [spk[key] for key in spk.keys() if len(spk[key]) > timestep]
            nutts = len(utts)
            if nutts > 2:
                k, l = 0, 0
                kidx = np.random.permutation(nutts)
                while l < batch_spk_samples:
                    utt = utts[kidx[k] % nutts]
                    k = k + 1 if k < nutts - 1 else 0
                    i = np.random.randint(0, len(utt) - timestep + 1)
                    feat.append(utt[i:i + timestep])
                    label.append(idx)
                    l += 1
                idx += 1
            if len(feat) == batch_size:
                feat = np.stack(feat)
                label = np.asarray(label)

                spks = np.zeros((batch_size, batch_nspks))
                spks[np.arange(batch_size), label] = 1.
                yield ({'features': feat[..., np.newaxis], 'labels': spks}, None)
                feat, label, idx = [], [], 0


def kaldi_data_generator(spkidxs, timestep=500, st=0, ed=1000, batch_nspks=None, batch_spk_samples=None, clean=False, seed=0):
    batch_size = batch_nspks * batch_spk_samples
    tspks, feats = {}, {}
    feats_path = '../sre16/v2/data/swbd_sre_combined_no_sil/feats.scp'
    spk2utt = '../sre16/v2/data/swbd_sre_combined_no_sil/spk2utt'

    with open(spk2utt) as f:
        for line in f.readlines():
            utts = line.decode().strip().split(' ')
            tspks[utts[0]] = utts[1:]

    fd = open_or_fd(feats_path)
    try:
        for line in fd:
            key, rxfile = line.decode().split(' ')
            feats[key] = rxfile
    finally:
        if fd is not feats_path : fd.close()

    kspks = tspks.keys()[st:ed]
    nspks = len(kspks)
    np.random.seed(seed)

    while True:
        feat, label, idx = [], [], 0
        for i in range(nspks):
            for spkidx in spkidxs[i][:batch_nspks]:
                utts = tspks[kspks[spkidx]]
                if clean:
                    utts = [utt for utt in utts if utt.split('-')[-1] not in ['music', 'reverb', 'babble', 'noise']]
                nutts = len(utts)

                k, l = 0, 0
                kidx = np.random.permutation(nutts)[:batch_spk_samples]
                nutts = np.minimum(nutts, batch_spk_samples)
                utts = [utts[i] for i in kidx]                
                futts = [read_mat(feats[utt]) for utt in utts]
                while l < batch_spk_samples:
                    utt = utts[k % nutts]
                    futt = futts[k % nutts]
                    k = k + 1 if k < nutts - 1 else 0
                    if len(futt) < timestep:
                        continue
                    i = np.random.randint(0, len(futt) - timestep + 1)
                    feat.append(futt[i:i + timestep])
                    label.append(idx)
                    l += 1
                idx += 1
            feat = np.stack(feat)
            label = np.asarray(label)

            spks = np.zeros((batch_size, batch_nspks))
            spks[np.arange(batch_size), label] = 1.
            yield ({'features': feat, 'labels': spks}, None)
            feat, label, idx = [], [], 0


def spk_sort(generator, timestep=500, st=0, ed=1000, batch_nspks=None, batch_spk_samples=None, clean=False, seed=0):
    batch_size = batch_nspks * batch_spk_samples
    tspks, feats = {}, {}
    feats_path = '../sre16/v2/data/swbd_sre_combined_no_sil/feats.scp'
    spk2utt = '../sre16/v2/data/swbd_sre_combined_no_sil/spk2utt'

    with open(spk2utt) as f:
        for line in f.readlines():
            utts = line.decode().strip().split(' ')
            tspks[utts[0]] = utts[1:]

    fd = open_or_fd(feats_path)
    try:
        for line in fd:
            key, rxfile = line.decode().split(' ')
            feats[key] = rxfile
    finally:
        if fd is not feats_path : fd.close()

    kspks = tspks.keys()[st:ed]
    nspks = len(kspks)
    np.random.seed(seed)

    X = []
    for idx, spk in enumerate(kspks):
        utts = tspks[spk]
        if clean:
            utts = [utt for utt in utts if utt.split('-')[-1] not in ['music', 'reverb', 'babble', 'noise']]
        
        for utt in utts:
            futt = read_mat(feats[utt])
            if len(futt) > timestep:
                i = np.random.randint(0, len(futt) - timestep + 1)
                X.append(futt[i:i + timestep])
                break

    X = np.stack(X)
    X = generator.predict(X, batch_size=128)
    scores = np.dot(X, X.T)
    spkidxs = np.argsort(scores)[:,::-1]
    return spkidxs


def kaldi_data(clean=False, path=None, min_dur=501):
    feats_path = path + "feats.scp"
    vad_path = path + "vad.scp"
    fd = open_or_fd(feats_path)
    feats = {}
    try:
        for line in fd:
            key, rxfile = line.decode().split(' ')
            feats[key] = rxfile
    finally:
        if fd is not feats_path : fd.close()

    fd = open_or_fd(vad_path)
    vads = {}
    try:
        for line in fd:
            key, rxfile = line.decode().split(' ')
            vads[key] = rxfile
    finally:
        if fd is not vad_path : fd.close()

    for key in feats:
        if clean and key.split('-')[-1] in ['music', 'reverb', 'babble', 'noise']:
            continue
        
        feat = read_mat(feats[key])
        vad = read_vec_flt(vads[key]).astype(bool)
        if np.sum(vad) > min_dur:
            cep_sliding_norm(feat, win=301, label=vad, center=True, reduce=False)
            yield (feat[vad], key)


def evaluate_XY(generator, st, ed, timestep=1000):
    print("Evaluate XY")
    X = kaldi_data_generator_bak(
        st=st, ed=ed,
        timestep=timestep, batch_nspks=200, batch_spk_samples=5, clean=True, seed=100).next()
    X_test = generator.predict(X[0]['features'], batch_size=32)
    Y_test = X[0]['labels']

    N = len(Y_test)
    print("Compute score")
    scores = np.dot(X_test, X_test.T)
    scores = scores.ravel()[np.eye(N).ravel() == 0]
    labels = np.dot(Y_test, Y_test.T)
    labels = labels.ravel()[np.eye(N).ravel() == 0]

    plotROC(scores, labels)


def plotROC(scores, labels, epoch=0, seed=1):
    fpr, fnr, thresholds, eer = det_curve(
        labels.ravel(), scores.ravel(), distances=False)

    # SRE-2008 performance parameters
    Cmiss = 10
    Cfa = 1
    P_tgt = 0.01
    Cdet08 = Cmiss * fnr * P_tgt + Cfa * fpr * (1 - P_tgt)
    dcf08 = 10 * np.min(Cdet08)

    # SRE-2010 performance parameters
    Cmiss = 1
    Cfa = 1
    P_tgt = 0.001
    Cdet10 = Cmiss * fnr * P_tgt + Cfa * fpr * (1 - P_tgt)
    dcf10 = 1000 * np.min(Cdet10)

    fig = plt.figure(figsize=(12, 12))
    plt.loglog(fpr, fnr, color='darkorange', lw=2, label='EER = %0.2f' % eer)
    print('EER = {0:.2f}%, THR = {1:.6f}'.format(
        eer * 100., thresholds[np.argmin(np.abs(fpr-eer))]))

    print('minDCF08 = {0:.4f}, THR = {1:.6f}'.format(
        dcf08, thresholds[np.argmin(Cdet08)]))
    print('minFPR = {0:.4f}%, minFNR = {1:.4f}%\n'.format(
        fpr[np.argmin(Cdet08)] * 100, fnr[np.argmin(Cdet08)] * 100))

    print('minDCF10 = {0:.4f}, THR = {1:.6f}'.format(
        dcf10, thresholds[np.argmin(Cdet10)]))
    print('minFPR = {0:.4f}%, minFNR = {1:.4f}%\n'.format(
        fpr[np.argmin(Cdet10)] * 100, fnr[np.argmin(Cdet10)] * 100))

    plt.loglog([eer], [eer], 'bo')
    plt.loglog([fpr[np.argmin(Cdet08)]], [fnr[np.argmin(Cdet08)]], 'ro')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.xlim(1e-4, 1.)
    plt.ylim(1e-2, 1.)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('./images/DET_' + str(epoch) + '.png', dpi=300, orientation='portrait')
    fig.savefig('./images/DET_latest.png', dpi=300, orientation='portrait')
    plt.close()


def evaluateSRE10(generator):
    print("Evaluating SRE10 10sec-10sec")
    dataset = h5py.File("./data/dataset_sre10.h5", "r")
    utts = dataset["sre10"].keys()

    X = []
    for utt in utts:
        X.append(dataset["sre10"].get(utt).value[:500])
    X = np.asarray(X)
    X_test = generator.predict(X[..., np.newaxis], batch_size=512)

    trials_path = "./local/SRE10SEC/trials"
    trials = np.loadtxt(
        trials_path,
        dtype={'names': ('enrol', 'test', 'key'), 'formats': ('S7', 'S7', 'S3')})

    enrol_idx = ismember(trials['enrol'], utts)
    test_idx = ismember(trials['test'], utts)
    labels = trials['key'] == 'tar'

    print('Scoring...')
    scores = np.sum(X_test[enrol_idx] * X_test[test_idx], axis=-1, keepdims=True)

    plotROC(scores, labels)


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]


train_model(niter=400)
