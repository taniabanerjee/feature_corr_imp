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
        if (abs(r) < 0.9):
            missed = missed + 1
        else:
            plist.append(x.index[i])
            y_pred[x.index[i]] = ccdf.loc[x.index[i]]['cluster.ID']
    missdf = missdf.drop(plist).copy()
    return missed, y_pred, missdf

def rf_body(cdf, train_index, test_index, features, classes):
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
    
    n_classes = len(classes)
    n_comp = min(len(features), n_classes-1)
    classifier = RandomForestClassifier(max_depth=n_comp, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #plot_result(ccdf_test, y_pred, cdf)
    
    acc = accuracy_score(y_test, y_pred)
    ccdf_test['y_pred'] = y_pred
    missdf = ccdf_test.iloc[np.where((y_pred-y_test) != 0)[0]]

    exptdf = pd.concat([ccdf_test, ccdf_train])
    delta = 0.025
    missed, y_pred2, missdf = compute_accuracy(ccdf, exptdf, missdf, delta, classes)
    y_pred = y_pred2[test_index]
    macc = accuracy_score(y_test, y_pred)

    return acc, macc

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
    delta = 0.025
    missed, y_pred2, missdf = compute_accuracy(ccdf, exptdf, missdf, delta, classes)
    y_pred = y_pred2[test_index]
    macc = accuracy_score(y_test, y_pred)

    return acc, macc

def lda_rf_body(cdf, train_index, test_index, features, classes):
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
    X_test = lda.transform(X_test)

    classifier = RandomForestClassifier(max_depth=n_comp, random_state=0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ccdf_test['y_pred'] = y_pred
    missdf = ccdf_test.iloc[np.where((y_pred-y_test) != 0)[0]]

    exptdf = pd.concat([ccdf_test, ccdf_train])
    delta = 0.025
    missed, y_pred2, missdf = compute_accuracy(ccdf, exptdf, missdf, delta, classes)
    y_pred = y_pred2[test_index]
    macc = accuracy_score(y_test, y_pred)

    return acc, macc

def getAvg(li):
    return sum(li)/len(li)

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
            acc2, macc2 = rf_body(ccdf, train_index, test_index, features, classes)
            acc3, macc3 = lda_rf_body(ccdf, train_index, test_index, features, classes)
            accuracy_lda.append(acc1)
            modified_accuracy_lda.append(macc1)
            accuracy_rf.append(acc2)
            modified_accuracy_rf.append(macc2)
            accuracy_lda_rf.append(acc3)
            modified_accuracy_lda_rf.append(macc3)
    
    print ('lda', len(features), getAvg(accuracy_lda), getAvg(modified_accuracy_lda))
    print ('rf', len(features), getAvg(accuracy_rf), getAvg(modified_accuracy_rf))
    print ('lda+rf', len(features), getAvg(accuracy_lda_rf), getAvg(modified_accuracy_lda_rf))

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
    run_models(ccdf, features1, classes)
    run_models(ccdf, features2, classes)
    run_models(ccdf, features3, classes)
    run_models(ccdf, features4, classes)

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
    features4 = ['tau-omega-2', 'lambda_4', 'C_1', 'C_2', 'S-omega-1', 'eta_3', 'tau-omega-1', 'lambda_2', 'S-tau-1', 'C_3', 'lambda_1']

    classes = [1, 2, 4, 6, 7, 11, 12, 13, 14, 15, 16]
    print ('channel flow with bump')
    #run_models(ccdf, features1, classes)
    #run_models(ccdf, features2, classes)
    #run_models(ccdf, features3, classes)
    run_models(ccdf, features4, classes)

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
    run_models(ccdf, features2, classes)
    run_models(ccdf, features3, classes)
    run_models(ccdf, features4, classes)

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
    features4 = ['lambda_2', 'S-tau-3', 'eta_5', 'lambda_4', 'eta_2', 'S-tau-2', 'eta_3', 'S-tau-1', 'lambda_1', 'C_1', 'C_2', 'C_3']
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19]
    print ('square cylinder')
    run_models(ccdf, features1, classes)
    run_models(ccdf, features2, classes)
    run_models(ccdf, features3, classes)
    run_models(ccdf, features4, classes)

#channel_flow()
#wavy_wall()
channel_with_bump()
#square_cyl()
