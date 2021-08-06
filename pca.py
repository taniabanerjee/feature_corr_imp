import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import random
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
    

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_coefs(mat, features, classes):
    #Data
    r = classes
    df = pd.DataFrame(mat)
    cols = get_cmap(20)

    #From raw value to percentage
    totals = np.sum(mat, axis=0)

    #plot
    barWidth = 0.85
    names = [str(i) for i in classes]
    i=0
    bars = [k/j * 100 for k,j in zip(mat[i], totals)]
    b = bars
    plt.bar(r, bars, color=cols(i), width=barWidth, label=features[i])
    print ('totals', totals)
    print (i, b)
    print ("mat", mat[i])
    print ("bars", bars)
    for i in range(1, mat.shape[0]):
        bars = [k/j * 100 for k,j in zip(mat[i], totals)]
        plt.bar(r, bars, bottom=b, color=cols(i), width=barWidth, label=features[i])
        b = [k+j for k,j in zip(b, bars)]
        print (i, b)
        print ("mat", mat[i])
        print ("bars", bars)
    plt.xticks(r, names)
    plt.xlabel('class')
    #plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.show()

def plot_result(ccdf_test, y_pred, cdf):
    fig = plt.figure()
    ax = fig.gca()
    yp = ccdf_test['y_plus']
    y_test = ccdf_test['cluster.ID']
    scatter = ax.scatter(yp, y_pred, c=y_test, cmap='tab10', s=5)
    plt.xscale('log')
    plt.xlim([1, 5000])
    plt.ylim([0, 6])
    plt.xlabel('$y^{+}$')
    plt.ylabel('Cluster')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
def plot_2d(X, y):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    y = y.astype(int)
    colors = get_cmap(19)
    plt.scatter(components[:, 0], components[:, 1], c=colors(y%19), label=y, s=5)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.show()

def plot_3d(X, y):
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    y = y.astype(int)
    colors = get_cmap(19)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(components[:, 0], components[:, 1], components[:, 2], c=colors(y%19), label=y, s=5)
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.set_zlabel('3rd principal component')
    plt.show()

def plot_error_reconstruction(scaled_data, max_comp):
    from numpy import linalg as LA
    pca = PCA(n_components=max_comp)
    pca2_results = pca.fit_transform(scaled_data)
    cumsum_variance = pca.explained_variance_ratio_.cumsum().copy()
    cumsum_variance = cumsum_variance * 100
    start=1
    error_record=[]
    for i in range(start,max_comp+1):
        pca = PCA(n_components=i)
        pca2_results = pca.fit_transform(scaled_data)
        pca2_proj_back=pca.inverse_transform(pca2_results)
        total_loss=LA.norm((scaled_data-pca2_proj_back),None)
        error_record.append(total_loss)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(error_record, 'o-', color=color)
    ax1.set_ylabel('reconstruction error', color=color)
    ax1.set_xlabel('pca components')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('variance', color=color)
    ax2.plot(cumsum_variance,'*-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.plot(cumsum_variance,'b*-')
    plt.xticks(range(len(error_record)), range(start,max_comp+1), rotation='vertical')
    plt.xlim([-1, len(error_record)])
    ax1.grid(b=True, which='major', color='#666666', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

def plot_pca(df, features, classes):
    #scale data
    X = df.loc[:, features].values
    y = df.loc[:, 'cluster.ID'].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    plot_2d(X, y)
    plot_3d(X, y)
    plot_error_reconstruction(X, min(len(classes), len(features)))
    return

def channel_flow():
    df = pd.read_csv('../../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../../toUFl.20210301/allChannel_features.dat', header=None, names=['y_plus', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'velocity_y', 'case.no'], delim_whitespace=True)

    ccdf = cdf.copy()
    ccdf['cluster.ID'] = df['cluster.ID']
    ccdf = ccdf.sort_values(by=['y_plus'])

    features1 = ['C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'lambda_1', 'eta_1', 'eta_3', 'eta_4']
    features3 = ['C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_2']
    features4 = ['lambda_5', 'C_2', 'eta_2', 'lambda_1', 'C_3']

    classes = [1, 2, 3, 4, 5]
    print ('channel flow')
    plot_pca(ccdf, features1, classes)
    plot_pca(ccdf, features2, classes)
    #plot_pca(ccdf, features3, classes)
    #plot_pca(ccdf, features4, classes)

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
    ccdf = cdf.copy()

    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'lambda_3']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_4', 'eta_3']
    features4 = ['C_2', 'S-omega-1', 'eta_3', 'tau-omega-1', 'lambda_2', 'S-tau-1', 'C_3', 'lambda_1']
    #features4 = ['tau-omega-2', 'lambda_4', 'C_1', 'C_2', 'S-omega-1', 'eta_3', 'tau-omega-1', 'lambda_2', 'S-tau-1', 'C_3', 'lambda_1']

    classes = [1, 2, 4, 6, 7, 11, 12, 13, 14, 15, 16]
    print ('channel flow with bump')
    plot_pca(ccdf, features1, classes)
    plot_pca(ccdf, features2, classes)
    #plot_pca(ccdf, features3, classes)
    #plot_pca(ccdf, features4, classes)

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
    plot_pca(ccdf, features1, classes)
    plot_pca(ccdf, features2, classes)
    #plot_pca(ccdf, features3, classes)
    #plot_pca(ccdf, features4, classes)

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

    ccdf = cdf.copy()
    features1 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5']
    features2 = ['C_1', 'C_2', 'C_3', 'S-tau-1']
    features3 = ['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'lambda_1', 'lambda_2', 'lambda_4', 'eta_2', 'eta_3', 'eta_5']
    features4 = ['eta_2', 'S-tau-2', 'eta_3', 'S-tau-1', 'lambda_1', 'C_1', 'C_2', 'C_3']
    #features4 = ['lambda_2', 'S-tau-3', 'eta_5', 'lambda_4', 'eta_2', 'S-tau-2', 'eta_3', 'S-tau-1', 'lambda_1', 'C_1', 'C_2', 'C_3']
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19]
    print ('square cylinder')
    plot_pca(ccdf, features1, classes)
    plot_pca(ccdf, features2, classes)
    #plot_pca(ccdf, features3, classes)
    #plot_pca(ccdf, features4, classes)

channel_flow()
wavy_wall()
channel_with_bump()
square_cyl()
