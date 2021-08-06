#!/usr/bin/env python
# coding: utf-8

# In[29]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#get_ipython().run_line_magic('matplotlib', 'inline')



 #In[30]:


#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = Name of the final output image after applying CRF
#Use_2d = boolean variable 
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import pydensecrf.densecrf as dcrf
from skimage import color

def crf(original_image, annotated_image,output_image, use_2d = True):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape)<3):
        annotated_image = color.gray2rgb(annotated_image).astype(np.uint32)
    
    annotated_image = annotated_image.astype(np.uint32)
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0].astype(np.uint32) + (annotated_image[:,:,1]<<8).astype(np.uint32) + (annotated_image[:,:,2]<<16).astype(np.uint32)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
#     print("No of labels in the Image are ")
#     print(n_labels)
    
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
#         d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
#                            compat=10,
#                            kernel=dcrf.DIAG_KERNEL,
#                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
#     cv2.imwrite(output_image, MAP.reshape(original_image.shape))
    x = MAP.reshape(24703, 3, 3)
    return x[:,:,1]
    #return MAP.reshape(original_image.shape)

# test = crf(frame, fgmask,'test.jpg', use_2d = True)
# plt.imshow(fgmask)
# plt.figure()
# plt.imshow(test)


# In[31]:

def get_indexes_from_cluster(ccdf, cid, pct):
    c1index = ccdf.index[ccdf['cluster.ID'] == cid].tolist()
    train1_index = random.sample(c1index, int(len(c1index)*pct))
    if (c1index[0] not in train1_index):
        train1_index.append(c1index[0])
    if (c1index[len(c1index)-1] not in train1_index):
        train1_index.append(c1index[len(c1index)-1])
    test1_index = [i for i in c1index if i not in train1_index]
    return train1_index, test1_index

def get_full_indexes(ccdf, classes, pct):
    train_index = []
    test_index = []
    for c in classes:
        tr, te = get_indexes_from_cluster(ccdf, c, pct)
        train_index = train_index + tr
        test_index = test_index + te

    return train_index, test_index

def get_vec_from_sphere(ccdf, delta, x, y, classes):
    n_classes = len(classes)
    points = ccdf[((np.square(ccdf['xB']-x) + np.square(ccdf['yB']-y)) < delta*delta)]
    vec = np.zeros(n_classes)
    for i in range(n_classes):
        vec[i] = points[points['y_pred']== classes[i]].shape[0]
    return vec

def compute_accuracy(ccdf, exptdf, missdf, delta, classes):
    x = missdf['xB']
    y = missdf['yB']
    rlist = []
    plist = []
    missed = 0
    y_pred = exptdf.y_pred.copy()
    for i in range(missdf.shape[0]):
        px = x.iloc[i]
        py = y.iloc[i]
        vec1 = get_vec_from_sphere(ccdf, delta, px, py, classes)
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = get_vec_from_sphere(exptdf, delta, px, py, classes)
        vec2 = vec2/np.linalg.norm(vec2)
        (r, p) = pearsonr(vec1,vec2)
        rlist.append(r)
        if (r < 0.99):
            missed = missed + 1
        else:
            plist.append(x.index[i])
            y_pred[x.index[i]] = ccdf.loc[x.index[i]]['cluster.ID']
    missdf = missdf.drop(plist).copy()
    return missed, y_pred, missdf

def lda_body(cdf, train_index, test_index, features, classes):
    ccdf = cdf.copy()
    ccdf['y_pred'] = ccdf['cluster.ID']
    ccdf_train = ccdf.loc[train_index]
    ccdf_test = ccdf.loc[test_index]
    X_train = ccdf_train.loc[:, features].values
    y_train = ccdf_train.loc[:, 'cluster.ID'].values
    X_test = ccdf_test.loc[:, features].values
    y_test = ccdf_test.loc[:, 'cluster.ID'].values

    X = X_test.copy()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    n_classes = np.unique(y_train).shape[0]
    n_comp = min(len(features), n_classes-1)
    lda = LDA(n_components=n_comp)
    X_train = lda.fit_transform(X_train, y_train)

    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ccdf_test['y_pred'] = y_pred
    missdf = ccdf_test.iloc[np.where((y_pred-y_test) != 0)[0]]

    exptdf = pd.concat([ccdf_test, ccdf_train])
    frame = np.array([[x,y,0] for x,y in zip(exptdf.x, exptdf.y)])
    cy = np.concatenate((y_pred, y_train))
    fgmask = np.array([(x, 0, 0) for x in cy])
    exptdf_filtered = crf(frame, fgmask,'', use_2d = True)
    y = exptdf_filtered[:,0]
    y_pred_c = y[0:y_pred.shape[0]]
    pts_test = np.where((y_pred_c-y_pred) != 0)[0].shape
    pts = np.where((y-cy) != 0)[0].shape
    delta = 0.025
    missed, y_pred2, missdf = compute_accuracy(ccdf, exptdf, missdf, delta, classes)
    missed_c, y_pred2_c, missdf_c = compute_accuracy(ccdf, exptdf, missdf, delta, classes)
    y_pred = y_pred2[test_index]
    macc = accuracy_score(y_test, y_pred)

    return acc, macc

def run_models(ccdf, features, classes):
    pct = [0.8]
    repeat_num = 20
    accuracy_lda = []
    modified_accuracy_lda = []
    accuracy_rf = []
    modified_accuracy_rf = []
    accuracy_lda_rf = []
    modified_accuracy_lda_rf = []
    for r in range(repeat_num):
        for p in pct:
            train_index, test_index = get_full_indexes(ccdf, classes, p)
            acc1, macc1 = lda_body(ccdf, train_index, test_index, features, classes)
            #acc2, macc2 = rf_body(ccdf, train_index, test_index, features, classes)
            #acc3, macc3 = lda_rf_body(ccdf, train_index, test_index, features, classes)
            accuracy_lda.append(acc1)
            modified_accuracy_lda.append(macc1)
            #accuracy_rf.append(acc2)
            #modified_accuracy_rf.append(macc2)
            #accuracy_lda_rf.append(acc3)
            #modified_accuracy_lda_rf.append(macc3)

    print ('LDA', len(features), getAvg(accuracy_lda), getAvg(modified_accuracy_lda))
    #print ('RF', len(features), getAvg(accuracy_rf), getAvg(modified_accuracy_rf))
    #print ('LDA->RF', len(features), getAvg(accuracy_lda_rf), getAvg(modified_accuracy_lda_rf))

def wavy_wall():
    df = pd.read_csv('../../toUFl.20210222/wavy.wall/data/wavywall_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../../toUFl.20210301/wavyWall_26features.dat', header=None, names=['x', 'y', 'z', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'flow.no'])
    val = []
    for index, row in cdf.iterrows():
        clus = df[(df['C_1']==row['C_1']) & (df['C_2']==row['C_2']) & \
                (df['C_3']==row['C_3']) & (df['eta_1']==row['eta_1'])]
        if (clus.shape[0] == 1):
            val.append(clus['cluster.ID'].item())
        elif (clus.shape[0] > 1):
            val.append(clus[0]['cluster.ID'].item())
        else:
            val.append(1001)
    cdf['cluster.ID'] = val
    cdf.drop(cdf[cdf['cluster.ID'] == 1001].index, inplace = True)

    ccdf = cdf.copy()
    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'eta_1']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'eta_2', 'eta_4', 'eta_5']
    features4 = ['eta_5', 'lambda_2', 'eta_2', 'lambda_1', 'C_1', 'S-tau-1', 'C_3', 'C_2']

    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print ('wavy wall')
    run_models(ccdf, features1, classes)
    return

wavy_wall()
