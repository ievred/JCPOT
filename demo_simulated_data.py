# -*- coding: utf-8 -*-
"""
Demo AISTATS
@author: Ievgen REDKO
"""

# !/usr/bin/env python


# %% Initialization

import matplotlib
from sklearn import svm,linear_model

import json

matplotlib.use('Agg')
import os
import pylab as pl
import numpy as np
import estimproportion as prop
import sklearn
import matplotlib.pyplot as plt
import time

# import seaborn as sns

plt.ioff()
np.random.seed(1976)
np.set_printoptions(precision=3)


def generateBiModGaussian(centers, sigma, n, A, b, h):
    result = {}
    xtmp = np.concatenate((centers[0] + sigma * np.random.standard_normal((int(h[0] * n), 2)),
                           centers[1] + sigma * np.random.standard_normal((int(h[1] * n), 2))))
    result['X'] = xtmp.dot(A) + b
    result['y'] = np.concatenate((np.zeros(int(h[0] * n)), np.ones(int(h[1] * n))))
    return result

# %% Dataset generation

# nb_source+1 domains : nb_source src + 1 tgt
# all domains have two classes
# proportions for each class are generated randomly

# %% Dataset generation
ns = 500

m1 = [-1, 0]
m2 = [1, 0]

A = np.eye(2)
# A=np.array([[2,1],[2,-1]])

nb_sources = 20
h_target = [0.2, 0.8]
all_Xr = []
all_Yr = []
b = [0, 0]
sigma = 1

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data_prior_shift/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for i in range(nb_sources):
    prop1 = np.random.randint(1, 99) / 100.0
    h_source = [prop1, round(1. - prop1, 2)]
    # bs = [b[np.random.randint(0,2)], b[np.random.randint(0,2)]]
    src = generateBiModGaussian([m1, m2], sigma, ns, A, b, h_source)
    all_Xr.append(src['X'])
    all_Yr.append(src['y'].astype(int))

n = 400

tgt = generateBiModGaussian([m1, m2], sigma, n, A, b, h_target)

# %%
print 'Distribution estimation'

Xt = tgt['X']
yt = tgt['y'].tolist()

possible_reg = np.logspace(-4, 1, 10)
possible_eta = np.logspace(-4, 1, 10)
sources = []
nb_tr = 5
step = 3


for j in xrange(2, len(all_Xr) + 1, step):
    print 'Number of sources is ' + str(j)
    sources.append(j)

    for i in range(nb_tr):
        reg, _, res_prop_na = prop.trueLODO(all_Xr[:j], all_Yr[:j], possible_reg, [0], 'acc', k=1)
        h_res, log = prop.estimateDensityBregmanProjection(all_Xr[:j], all_Yr[:j], Xt, reg, numItermax=300)

        print 'Prop estimation accuracy=', float("{0:.4f}".format(prop.computeKLdiv(h_res, h_target)))

        estimatedy = prop.estimatekNN(all_Xr, all_Yr, Xt)
        aa = float(np.sum(tgt['y'] == estimatedy)) / len(tgt['y'])
        print ' 1NN Accuracy=', aa

        estimatedy = prop.estimateLabels(all_Yr, len(tgt['y']), log)
        aa = float(np.sum(tgt['y'] == estimatedy)) / len(tgt['y'])
        print ' (Label Prop reg) w/ proportion  Accuracy=', aa

        estimatedy = prop.estimateLabelsPoints(all_Yr, Xt, log, k=1)
        aa = float(np.sum(tgt['y'] == estimatedy)) / len(tgt['y'])
        print ' (Point transport reg) w/ proportion Accuracy=', aa

        log2 = prop.estimateTransport(all_Xr, Xt, reg, numItermax=100)

        estimatedy = prop.estimateLabels(all_Yr, len(tgt['y']), log2)
        aa = float(np.sum(tgt['y'] == estimatedy)) / len(tgt['y'])
        print ' (Label Prop) 1-1 transport Accuracy=', aa

        estimatedy = prop.estimateLabelsPoints(all_Yr, Xt, log2, k=1)
        aa = float(np.sum(tgt['y'] == estimatedy)) / len(tgt['y'])
        print ' (Point transport) 1-1 transport Accuracy=', aa

