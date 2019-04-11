# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:34:14 2015

@author: Nicolas Courty & Ievgen Redko
"""

import sys
from operator import itemgetter
import numpy as np
import random
import sklearn
import sklearn.neighbors
from sklearn import svm
from functools import reduce
from scipy.spatial.distance import cdist
import multiprocessing as mp
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

clf = SGDClassifier()
#clf = LinearSVC()

def computeKLdiv(h1,h2):
    if len(h1)!=len(h2):
        print('problem in KL divergence: different lengths for distributions')
        return 0
    else:
        divKL = 0
        for i in range(len(h1)):
            divKL = divKL + h1[i]*np.log(h1[i]/h2[i])
        return divKL

def computeL1div(h1,h2):
    if len(h1)!=len(h2):
        print('problem in L1 distance: different lengths for distributions')
        return 0
    else:
        divL1 = 0
        for i in range(len(h1)):
            divL1 = divL1 + np.fabs(h1[i]-h2[i])
        return divL1/len(h1)

def getProportions(Y,minlab=0,maxlab=9):
    return np.array([np.sum(Y==i)/float(len(Y)) for i in np.arange(minlab,maxlab+1)])

def geometricBar(weights,alldistribT):
    assert(len(weights)==alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT),weights.T))

def geometricMean(alldistribT):
    return np.exp(np.mean(np.log(alldistribT),axis=1))

def projR(gamma,p):
    #return np.dot(np.diag(p/np.maximum(np.sum(gamma,axis=1),1e-10)),gamma)
    return np.multiply(gamma.T,p/np.maximum(np.sum(gamma,axis=1),1e-10)).T

def projC(gamma,q):
    #return (np.dot(np.diag(q/np.maximum(np.sum(gamma,axis=0),1e-10)),gamma.T)).T
    return np.multiply(gamma,q/np.maximum(np.sum(gamma,axis=0),1e-10))

def estimateTransport(all_Xr,Xt,reg,numItermax = 100, tol_error=1e-7):
    nbdomains = len(all_Xr)
    # we then build, for each source domain, specific information
    all_domains = {}
    for d in range(nbdomains):
        all_domains[d] = {}
        # get number of elements for this domain
        nb_elem = all_Xr[d].shape[0]
        all_domains[d]['nbelem'] = nb_elem
        # build the distance matrix
        M = cdist(all_Xr[d],Xt,metric='sqeuclidean')
        M = M/np.median(M)
        K = np.exp(-M/reg)
        all_domains[d]['K'] = K
        all_domains[d]['w'] = np.ones(nb_elem).astype(float)/nb_elem

    distrib = np.ones(Xt.shape[0])/Xt.shape[0]

    cpt=0
    log = {}

    while (cpt<numItermax):
        for d in range(nbdomains):
            all_domains[d]['K'] = projC(all_domains[d]['K'],distrib)
            all_domains[d]['K'] = projR(all_domains[d]['K'],all_domains[d]['w'])
        cpt=cpt+1

    log['all_domains']=all_domains
    return log

def prepareDomains(all_Xr,all_Yr,Xt):
    if len(all_Yr) == 1:
        classes = np.unique(all_Yr[0])
    else:
        classes = reduce(np.union1d, all_Yr)
    nbclasses = len(classes)
    nbdomains = len(all_Xr)
    # we then build, for each source domain, specific information
    all_domains = []
    for d in range(nbdomains):
        dom = {}
        # get number of elements for this domain
        nb_elem = all_Xr[d].shape[0]
        dom['nbelem'] = nb_elem
        # build the corresponding D matrix
        D = np.zeros((nbclasses,nb_elem))
        D2 = np.zeros((nbclasses,nb_elem))
        classes_d = np.zeros(nbclasses)
        if np.min(np.unique(all_Yr[d]))!=0:
            all_Yr[d] = all_Yr[d] - np.min(np.unique(all_Yr[d]))
        classes_d[np.unique(all_Yr[d]).astype(int)]=1
        dom['classes']=classes_d
        for c in classes:
            nbelemperclass = np.sum(all_Yr[d]==c)
            if nbelemperclass!=0:
                D[int(c),all_Yr[d]==c]=1./(nbelemperclass)#*nbclasses_d)
                D2[int(c),all_Yr[d]==c]=1.
        dom['D'] = D
        dom['D2'] = D2

        # build the distance matrix
        M = cdist(all_Xr[d],Xt,metric='sqeuclidean')
        M = M/np.median(M)

        dom['K'] = np.zeros_like(M)
        dom['M'] = M
        dom['G'] = np.zeros_like(M)

        all_domains.append(dom)
    return np.array(all_domains), classes

def innerLoop(all_domains,all_Yr,Xt,classes,reg,eta,nSinkhornloop=100,numIterMM=10,tol_error=1e-7):
    distrib = np.ones(Xt.shape[0])/Xt.shape[0]
    nbdomains = len(all_domains)
    nbclasses=len(classes)
    cpt=0
    log = {'niter':0, 'all_delta':[]}
    p=0.5
    epsilon = 1e-3
    flagNan = 0
    for i in range(numIterMM):
        old_bary = np.ones(nbclasses)
        err=1
        inner_cpt=0
        #  perform majoration 
        for d in range(nbdomains):
            all_domains[d]['K'] = np.exp(-(all_domains[d]['M'] + eta * all_domains[d]['G'])/reg)
            if np.isnan(all_domains[d]['K']).any():
                flagNan = 1
                break
        if flagNan!=1:
            while (err>tol_error and inner_cpt<nSinkhornloop):
                bary = np.zeros((nbclasses))

                for d in range(nbdomains):
                    all_domains[d]['K'] = projC(all_domains[d]['K'],distrib)
                    other = np.sum(all_domains[d]['K'],axis=1)
                    bary = bary + np.log(np.dot(all_domains[d]['D2'],other)) / nbdomains
                bary = np.exp(bary)
                if np.isnan(bary).any():
                    print('Nans in bary. Quiting the InnerLoop.')
                    break
                for d in range(nbdomains):
                    new = np.dot(all_domains[d]['D'].T,bary)
                    all_domains[d]['K'] = projR(all_domains[d]['K'],new)

                err=np.linalg.norm(bary-old_bary)
                log['all_delta'].append(err)
                inner_cpt = inner_cpt+1
                old_bary=bary
            cpt=cpt+inner_cpt
            bary = bary / np.sum(bary)

            if eta!=0:
                #  compute norm by class
                for t in range(Xt.shape[0]):
                    for c in classes:
                        sum_c = 0
                        for d in range(nbdomains):
                            sum_c+=np.sum(all_domains[d]['K'][all_Yr[d]==c,t])
                        maj = p*((sum_c+epsilon)**(p-1))
                        for d in range(nbdomains):
                            all_domains[d]['G'][all_Yr[d]==c,t]=maj
        else:
            print('Nans in the Inner Loop matrix K')
            log['niter'] = cpt
            log['all_domains'] = all_domains
            return old_bary,log
    #print cpt
    log['niter']=cpt
    log['all_domains']=all_domains
    return bary,log

def estimateDensityBregmanProjection(all_Xr, all_Yr,Xt,reg,numItermax = 100, tol_error=1e-7):
    all_domains, classes = prepareDomains(all_Xr,all_Yr,Xt)
    return innerLoop(all_domains,all_Yr,Xt,classes,reg,eta=0,nSinkhornloop=numItermax,numIterMM=1,tol_error=tol_error)


def estimateDensityBregmanProjectionClassreg(all_Xr, all_Yr,Xt,reg,eta,h_target=None,numItermax = 100, numIterMM= 10, tol_error=1e-7):
    all_domains, classes = prepareDomains(all_Xr,all_Yr,Xt)
    return innerLoop(all_domains,all_Yr,Xt,classes,reg,eta,numItermax,numIterMM,tol_error)


#%% classification

def estimateLabels(all_Yr,nbxt,log):
    all_fused_Y=estimateProps(all_Yr,nbxt,log)
    return np.argmax(all_fused_Y,axis=1), all_fused_Y[:,1]

def estimateProps(all_Yr,nbxt,log):
    n = len(np.unique(reduce(np.union1d, all_Yr)))
    nd =len(log['all_domains'])
    all_fused_Y = np.zeros((nbxt,n))
    for d in range(nd):
        mat_lab_d = np.zeros((all_Yr[d].shape[0],n))
        for i in range(all_Yr[d].shape[0]):
            mat_lab_d[i,int(all_Yr[d][i])]=1
        transp = log['all_domains'][d]['K']
        transp= transp.shape[1]*transp
        all_fused_Y = all_fused_Y+np.dot(mat_lab_d.T,transp).T / nd

    return all_fused_Y

def estimatekNN(all_Xr,all_Yr,Xt,k=1):
    ## NN estimation
    learn_setX = all_Xr[0]
    learn_setY = all_Yr[0]
    for d in range(1,len(all_Xr)):
        learn_setX = np.concatenate((learn_setX,all_Xr[d]))
        learn_setY = np.concatenate((learn_setY,all_Yr[d]))
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    neigh.fit(learn_setX,learn_setY)
    return neigh.predict(Xt)

def estimateTranspPoints(Xt,log):
    nd =len(log['all_domains'])
    all_Xr_transp=[]
    for d in range(nd):
        transp = log['all_domains'][d]['K']
        transp1 = np.dot(np.diag(1/(np.sum(transp,1)+1e-8)),transp)
        all_Xr_transp.append(np.array(np.dot(transp1,Xt)))
    return all_Xr_transp

def estimateLabelsPoints(all_Yr,Xt,log,k=1):
    nd =len(log['all_domains'])
    nbxt = Xt.shape[0]
    all_Xr_transp=[]
    for d in range(nd):
        transp = log['all_domains'][d]['K']
        transp1 = np.dot(np.diag(1/(np.sum(transp,1)+1e-8)),transp)
        all_Xr_transp.append(np.array(np.dot(transp1,Xt)))

    return estimatekNN(all_Xr_transp,all_Yr,Xt,k)

def LODO(all_Xr, all_Yr,Xt,Yt,possible_reg,possible_eta,k=1):
    sizer = len(possible_reg)
    sizee = len(possible_eta)
    results = np.zeros((sizer,sizee))
    results_test = np.zeros((sizer,sizee))
    nbdomains = len(all_Xr)
    all_Yr =  np.array(all_Yr)

    all_domains, classes = prepareDomains(all_Xr,all_Yr,Xt)

    for r,i in zip(possible_reg,range(sizer)):
        for e,j in zip(possible_eta,range(sizee)):
            if e==0:
                h,log = innerLoop(all_domains,all_Yr,Xt,classes,r,e,nSinkhornloop=100,numIterMM=1,tol_error=1e-7)
            else:
                h,log = innerLoop(all_domains,all_Yr,Xt,classes,r,e,nSinkhornloop=100,numIterMM=10,tol_error=1e-7)

            all_newXr = np.array(estimateTranspPoints(Xt,log))
            for d in range(nbdomains):
                dom = range(nbdomains)
                dom.remove(d)
                dom = np.array(dom)
                estimatedy = estimatekNN(all_newXr[dom],all_Yr[dom],all_newXr[d],k)
                perf = float(np.sum(all_Yr[d]==estimatedy))/len(all_Yr[d])
                results[i,j]+=perf

            estimatedy = estimateLabelsPoints(all_Yr,Xt,log,k)[0]
            results_test[i,j]=float(np.sum(Yt==estimatedy))/len(Yt)
            #print 'on test : ', float(np.sum(Yt==estimatedy))/len(Yt)

    bestr,beste = np.unravel_index(results.argmax(), results.shape)
    return possible_reg[bestr],possible_eta[beste]

def trueLODO(all_Xr, all_Yr,possible_reg,possible_eta,method='labprop',k=1):
    sizer = len(possible_reg)
    sizee = len(possible_eta)
    results = np.zeros((sizer,sizee))

    #if len(all_Xr)<3:
    nbdomains = len(all_Xr)
    #else:
    #    nbdomains = 5
    all_Yr =  np.array(all_Yr)
    all_Xr =  np.array(all_Xr)

    for d in range(nbdomains):
        dom = list(range(nbdomains))
        if nbdomains > 1:
            dom.remove(d)

        dom = np.array(dom)
        #all_domains, classes = prepareDomains(all_Xr[dom],all_Yr[dom],all_Xr[d])
        for r,i in zip(possible_reg,range(sizer)):
            flagNan=0
            for e,j in zip(possible_eta,range(sizee)):
                all_domains, classes = prepareDomains(all_Xr[dom],all_Yr[dom],all_Xr[d])
                if e==0:
                    h,log = innerLoop(all_domains,all_Yr[dom],all_Xr[d],classes,r,e,nSinkhornloop=10,numIterMM=1,tol_error=1e-10)
                    for d1 in range(nbdomains-1):
                        if np.isnan(log['all_domains'][d1]['K']).any():
                            flagNan = 1
                            break
                    if np.isnan(h).any() or flagNan==1:
                        break
                else:
                    h,log = innerLoop(all_domains,all_Yr[dom],all_Xr[d],classes,r,e,nSinkhornloop=10,numIterMM=10,tol_error=1e-10)
                    if np.isnan(h).any():
                        break
                if method=='labprop':
                    estimatedy = estimateLabels(all_Yr[dom],len(all_Yr[d]),log)[0]
                    perf=float(np.sum(all_Yr[d]==estimatedy))/len(all_Yr[d])
                elif method == 'prop':
                    perf = computeL1div(h,getProportions(all_Yr[d],0,3))
                elif method == 'knn':
                    all_newXr = np.array(estimateTranspPoints(all_Xr[d],log))
                    estimatedy = estimatekNN(all_newXr,all_Yr[dom],all_Xr[d],k)
                    perf = float(np.sum(all_Yr[d]==estimatedy))/len(all_Yr[d])
                else:
                    all_newXr = np.concatenate(estimateTranspPoints(all_Xr[d], log))
                    perf = clf.fit(all_newXr, np.concatenate(all_Yr[dom])).score(all_Xr[d], all_Yr[d])
                results[i,j]+=perf

    #print results.T/nbdomains
    if method == 'prop':
        bestr, beste = np.unravel_index(results.argmin(), results.shape)
    else:
        bestr,beste = np.unravel_index(results.argmax(), results.shape)
    #print possible_reg[bestr],possible_eta[beste]
    return possible_reg[bestr],possible_eta[beste],results

def trueLODO_indep(all_Xr, all_Yr,possible_reg,method='labprop',k=1):
    sizer = len(possible_reg)
    results = np.zeros((sizer,1))
    nbdomains = len(all_Xr)
    all_Yr =  np.array(all_Yr)
    all_Xr =  np.array(all_Xr)


    for d in range(nbdomains):
        dom = range(nbdomains)
        if nbdomains>1:
            dom.remove(d)
        dom = np.array(dom)

        for r,i in zip(possible_reg,range(sizer)):
            log = estimateTransport(all_Xr[dom],all_Xr[d],r,numItermax = 10)
            if method=='labprop':
                estimatedy = estimateLabels(all_Yr[dom],len(all_Yr[d]),log)[0]
                perf=float(np.sum(all_Yr[d]==estimatedy))/len(all_Yr[d])
            else:
                all_newXr = np.array(estimateTranspPoints(all_Xr[d],log))
                estimatedy = estimatekNN(all_newXr,all_Yr[dom],all_Xr[d],k)
                perf = float(np.sum(all_Yr[d]==estimatedy))/len(all_Yr[d])
            results[i]+=perf
    # print results

    bestr,beste = np.unravel_index(results.argmax(), results.shape)
    return possible_reg[bestr]


def get_LODO_concat(all_Xr,all_Yr,i):
    xtest=all_Xr[i]
    ytest=all_Yr[i]
    xapp=reduce(lambda x,y : np.vstack((x,y)),[x for j,x in enumerate(all_Xr) if j!=i])
    yapp=reduce(lambda x,y : np.concatenate((x,y),0),[x for j,x in enumerate(all_Yr) if j!=i])
    return xapp,yapp,xtest,ytest
