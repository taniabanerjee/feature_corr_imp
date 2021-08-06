import numpy as np
import pandas as pd
import math
import sys
import datetime
import time
import random
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_sorted_cid_and_plot(df, fig, ax, n_clusters, cluster_map):
    n_points = df.shape[0]
    cluster_ids_unique = df['cluster.ID.new'].unique()
    print('Cluster statistics ...{} points, {} clusters\n'.format(n_points, n_clusters))
#    points = []
#    for cluster_id in cluster_ids_unique:
#        rows = df[df['cluster.ID'] == cluster_id]
#        print(' Cluster ID = {} has {} points'.format(cluster_id, len(rows)))
#        points.append(len(rows))
#    sorted_cid = [x for _,x in sorted(zip(points,cluster_ids_unique))]
    sorted_cid = sorted(cluster_ids_unique)
    ## ---- Find colors for clusters
    #cols = get_cmap(len(cluster_ids_unique))
    cols = get_cmap(19)
    for cluster_id in sorted_cid:
        rows = df[df['cluster.ID.new'] == cluster_id]
        orig_id = cluster_map[cluster_id]
        ax.scatter(rows['x'], rows['y'], s=5, color=cols(orig_id%19), label=str(orig_id))
    plt.legend(loc='right', title='Clusters')
    plt.xlabel('x')
    plt.ylabel('y')
    return

def get_cartesian_from_barycentric(b):
    t = np.transpose(np.array([[0,0],[1,0],[0.5,sqrt(3)/2]])) # Triangle
    return t.dot(b)

def barydist(ci, cj):
    p1 = ci[0]
    p2 = ci[1]
    p3 = ci[2]
    q1 = cj[0]
    q2 = cj[1]
    q3 = cj[2]
    return -(q2 - p2)*(q3 - p3) - (q1 - p1)*(q3-p3) - (q1 - p1)*(q2 - p2)

def distance(xi, xj, isb):
    if (isb):
        ci = [xi[0], xi[1], xi[2]]
        cj = [xj[0], xj[1], xj[2]]
        return np.sum(np.square(np.subtract(xi[3:], xj[3:]))) + barydist(ci, cj)
    else:
        return np.sum(np.square(np.subtract(xi, xj)))

def similarity(w, tau):
    #sigmalist.append(np.sqrt(w)/0.832555)
    return np.exp(-w/(2*tau*tau)), w/(2*tau*tau)

def plot_orig_clusters(df):
    classes = [1, 2, 3, 4, 5]
    names = {1: 'outer layer', 2:'outer buffer layer', 3:'inner buffer layer', 4:'log layer', 5:'viscous sublayer'}
    cluster_map = getMap(df, classes)
    fig = plt.figure()
    ax = fig.gca()
    cx = []
    cy = []
    textdict = {}
    for i in range (0, 5):
        y = df[df['cluster.ID.new'] == i]
        yp = y['y_plus']
        yy = [[i] * yp.size]
        cx = cx + list(yp)
        cy = cy + yy[0]
        index = int(yp.size/10)
        xloc = yp.iloc[0]+1
        if (xloc > 100):
            xloc = 100
        yloc = cluster_map[yy[0][index]]+0.2
        if (textdict.get(yloc) == None):
            plt.text(xloc, yloc, '{}, {}, {}, {}'.format(round(min(yp), 2), round(max(yp), 2), yp.size, names[cluster_map[i]]))
            textdict[yloc] = True
    colors = get_cmap(19)
    ones = [1]*len(cy)
    labels = [cluster_map[x] for x, y in zip(cy, ones)]
    scatter = ax.scatter(cx, labels, c=colors(labels), s=5)
    plt.xscale('log')
    plt.xlim([1, 5000])
    plt.ylim([0, 6])
    plt.xlabel('$y^{+}$')
    plt.ylabel('Cluster')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    missedStats(df, cluster_map)
    return

def missedStats(df, cluster_map):
    labels = [cluster_map[x] for x in df['cluster.ID.new']]
    missed = np.where(df['cluster.ID'] != labels)[0]
    print ('total points:', df.shape[0])
    print ('missed_points:', len(missed))
    print ('% correct:', (df.shape[0] - len(missed)) * 100/ df.shape[0])
    return

def findClusters(plotdf, features, classes, hasBaryCentricCoordinates, n_comp):
    tau = 1000
    eps = 0.3
    knn = 1
    target = plotdf[features]
    odf = StandardScaler().fit_transform(target)
    #df = odf[np.where(plotdf['case.no']==5)]
    #plotdf = plotdf[plotdf['case.no']==5]
    df = odf
    pca = PCA(n_components=4)
    df = pca.fit_transform(df)
    num_points = df.shape[0]
    W = np.zeros([num_points, num_points])
    WNN = np.zeros([num_points,num_points])
    #W = np.full([num_points, num_points], 1.0e+6, dtype=float)
    #WNN = np.full([num_points,num_points], 1.0e+6, dtype=float)
    print ('Compute the weight matrix')
    distlist = []
    simlist = []
    ratiolist = []
    for i in range(0, num_points):
        W[i,i] = 1
        for j in range (i+1, num_points):
            w = distance(df[i], df[j], hasBaryCentricCoordinates)
            #distlist.append(w)
            W[i, j], ratio = similarity(w, tau)
            W[j, i] = W[i, j]
            #simlist.append(W[i, j])
            #ratiolist.append(ratio)
            #if (w < eps): 
                #W[i, j] = 1
                #W[j, i] = W[i, j]
    '''
    print ('Compute the KNN neighbors')
    distarray = np.asarray(distlist)
    ddf = pd.DataFrame(distlist, columns=['distance'])
    ddf.to_csv('cfdist.csv')
    n, bins, patches = plt.hist(distarray)
    plt.show()
    simarray = np.asarray(simlist)
    n, bins, patches = plt.hist(simarray)
    plt.show()
    ratioarray = np.asarray(ratiolist)
    n, bins, patches = plt.hist(ratioarray)
    plt.show()

    WNN = W.copy()
    while (knn > 1):
        WNN = sp.linalg.blas.dgemm(1.0, WNN, W)
        knn = knn-1
    '''
    WNN = W.copy()
    print ('Compute the diagonal matrix degree matrix')
    D = np.zeros([num_points,num_points])
    for i in range(0, num_points):
        D[i,i] = np.sum(WNN[i,:])
    Dinv = np.linalg.inv(D)
    
    print ('Compute the Laplacian matrix L')
    L = D-WNN
    print ('Compute Dinverse times L')
    #A = np.einsum('ij,jk->ik', Dinv, L)
    A = sp.linalg.blas.dgemm(1.0, Dinv, L)
    
    print ('Compute the eigenvalues and eigenvectors')
    #vals, vecs = sp.linalg.eig(A)
    vals, vecs = sp.linalg.eig(A)
    vecs = vecs[:,np.argsort(vals)].real
    vals = vals[np.argsort(vals)].real
    
    print ('Compute Kmeans clustering')
    num_clusters=len(classes)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vecs[:,1:num_clusters+1])
    y_labels = kmeans.labels_
    plotdf['cluster.ID.new'] = y_labels

    return y_labels

def plot_embedding(df):
    reducer = umap.UMAP(random_state=42)
    scaled_data = StandardScaler().fit_transform(df)
    embedding = reducer.fit_transform(scaled_data)
    print (embedding.shape)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.subplot(2, 1, 1)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_title('Original, using GMM')
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in df['cluster.ID']])
    plt.subplot(2, 1, 2)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_title('Using Spectral Embedding')
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in df['new_label']])
    plt.show()

def getMap(df, classes):
    import operator
    cluster_ids_unique = df['cluster.ID.new'].unique()
    cluster_map = {}
    num = {}
    for k in cluster_ids_unique:
        c = df[df['cluster.ID.new'] == k]
        for i in classes:
            num[i] = c[c['cluster.ID'] == i].shape[0]
        cluster_map[k] = max(num.items(), key=operator.itemgetter(1))[0]
    return cluster_map

def plot_complex(df, classes, n_clusters):
    cluster_map = getMap(df, classes)
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax, n_clusters, cluster_map)
    plt.show()
    missedStats(df, cluster_map)
    return

def channel_flow():
    rdf = pd.read_csv('../../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    df = pd.read_csv('../../toUFl.20210301/allChannel_features.dat', header=None, names=['y_plus', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'velocity_y', 'case.no'], delim_whitespace=True)
    df['cluster.ID'] = rdf['cluster.ID']

    features1 = ['C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3']
    features3 = ['C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_2']
    features4 = ['lambda_5', 'C_2', 'eta_2', 'lambda_1', 'C_3']

    classes = [1, 2, 3, 4, 5]
#    df['cluster.ID.new'] = df['cluster.ID'] - 1
#    plot_orig_clusters(df)
    plotdf1 = findClusters(df, features2, classes, False, len(features2))
    plot_orig_clusters(df)
#    plotdf2 = findClusters(df, features2, classes, False)
#    plot_orig_clusters(df)
#    plotdf3 = findClusters(df, features3, classes, False)
#    plot_orig_clusters(df)
#    plotdf4 = findClusters(df, features4, classes, True, min(len(features2), len(features4)))
#    plot_orig_clusters(df)

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
    cdf.drop(cdf[cdf['cluster.ID'] == 1].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 2].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 3].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 4].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 5].index, inplace = True)

    ccdf = cdf.copy()
    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'eta_1']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'eta_2', 'eta_4', 'eta_5']
    features4 = ['C_1', 'C_2', 'C_3', 'eta_5', 'lambda_2', 'eta_2', 'lambda_1', 'S-tau-1']

    classes = [6, 7, 8, 9, 10]
    print ('wavy wall')
#    ccdf['cluster.ID.new'] = cdf['cluster.ID']
#    plot_complex(ccdf, classes, 10)
    ypred1 = findClusters(ccdf, features2, classes, False, len(features2))
    plot_complex(ccdf, classes, 10)
#    plotdf2 = findClusters(ccdf, features2, classes, False)
#    plot_complex(ccdf, classes, 10)
#    plotdf3 = findClusters(ccdf, features3, classes, False)
#    plot_complex(ccdf, classes, 10)
#    plotdf4 = findClusters(ccdf, features4, classes, False)
#    plot_complex(ccdf, classes, 10)

def channel_with_bump():
    df = pd.read_csv('../../toUFl.20210222/channel.with.bump/data/bump_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'cos-theta_S-tau', 'lambda_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../../toUFl.20210301/channelWithBump_26features.dat', header=None, names=['x', 'y', 'z', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'flow.no'])
    val = []
    for index, row in cdf.iterrows():
        clus = df[(df['C_1']==row['C_1']) & (df['C_2']==row['C_2']) & \
                (df['C_3']==row['C_3']) & (df['lambda_3']==row['lambda_3']) & \
                (df['cos-theta_S-tau']==row['S-tau-1'])]
        if (clus.shape[0] == 1):
            val.append(clus['cluster.ID'].item())
        elif (clus.shape[0] > 1):
            val.append(clus[0]['cluster.ID'].item())
        else:
            val.append(1001)
    cdf['cluster.ID'] = val
    cdf.drop(cdf[cdf['cluster.ID'] == 1001].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 1].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 2].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 4].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 6].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 7].index, inplace = True)
    ccdf = cdf.copy()

    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'lambda_3']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_4', 'eta_3']
    features4 = ['C_2', 'S-omega-1', 'eta_3', 'tau-omega-1', 'lambda_2', 'S-tau-1', 'C_3', 'lambda_1']
    #features4 = ['tau-omega-2', 'lambda_4', 'C_1', 'C_2', 'S-omega-1', 'eta_3', 'tau-omega-1', 'lambda_2', 'S-tau-1', 'C_3', 'lambda_1']

    classes = [11, 12, 13, 14, 15, 16]
    print ('channel flow with bump')
#    ccdf['cluster.ID.new'] = cdf['cluster.ID']
#    plot_complex(ccdf, classes, 11)
    ypred1 = findClusters(ccdf, features2, classes, False, len(features2))
    plot_complex(ccdf, classes, 11)
#    plotdf2 = findClusters(ccdf, features2, classes, False)
#    plot_complex(ccdf, classes, 11)
#    plotdf3 = findClusters(ccdf, features3, classes, False)
#    plot_complex(ccdf, classes, 11)
#    plotdf4 = findClusters(ccdf, features4, classes, True)
#    plot_complex(ccdf, classes, 11)

def square_cyl():
    df = pd.read_csv('../../toUFl.20210222/square.cylinder/data/sqcyl_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'cos theta_S-tau', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../../toUFl.20210301/squareCylinder_26features.dat', header=None, names=['x', 'y', 'z', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'flow.no'])
    val = []
    for index, row in cdf.iterrows():
        clus = df[(df['C_1']==row['C_1']) & (df['C_2']==row['C_2']) & \
                (df['C_3']==row['C_3']) & (df['cos theta_S-tau']==row['S-tau-1'])]
        if (clus.shape[0] == 1):
            val.append(clus['cluster.ID'].item())
        elif (clus.shape[0] > 1):
            val.append(clus.iloc[0]['cluster.ID'].item())
        else:
            val.append(1001)
    cdf['cluster.ID'] = val
    cdf.drop(cdf[cdf['cluster.ID'] == 1001].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 1].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 2].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 3].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 4].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 5].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 6].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 7].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 8].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 9].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 10].index, inplace = True)
    cdf.drop(cdf[cdf['cluster.ID'] == 11].index, inplace = True)

    ccdf = cdf.copy()
    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'S-tau-1']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'lambda_1', 'lambda_2', 'lambda_4', 'eta_2', 'eta_3', 'eta_5']
    features4 = ['eta_2', 'S-tau-2', 'eta_3', 'S-tau-1', 'lambda_1', 'C_1', 'C_2', 'C_3']
    #features4 = ['lambda_2', 'S-tau-3', 'eta_5', 'lambda_4', 'eta_2', 'S-tau-2', 'eta_3', 'S-tau-1', 'lambda_1', 'C_1', 'C_2', 'C_3']

    classes = [17, 18, 19]
    print ('square cylinder')
#    ccdf['cluster.ID.new'] = cdf['cluster.ID']
#    plot_complex(ccdf, classes, 16)
    ypred1 = findClusters(ccdf, features2, classes, False, len(features2))
    plot_complex(ccdf, classes, 16)
#    plotdf2 = findClusters(ccdf, features2, classes, False)
#    plot_complex(ccdf, classes, 16)
#    plotdf3 = findClusters(ccdf, features3, classes, False)
#    plot_complex(ccdf, classes, 16)
#    plotdf4 = findClusters(ccdf, features4, classes, True)
#    plot_complex(ccdf, classes, 16)

channel_flow()
wavy_wall()
channel_with_bump()
square_cyl()
