import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import random
from scipy.stats.stats import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

def find_selected_features(cdf, features):
    from scipy.cluster import hierarchy
    import seaborn as sns
    import heatmap
    sns.set()
    dfcorr = cdf[features].corr(method='pearson')
    
    plt.figure(figsize=(9, 9))
    heatmap.corrplot(dfcorr)
    plt.show()
    
    corr = dfcorr.to_numpy()
    
    corr_linkage = hierarchy.ward(corr)
    plt.figure(figsize=(9, 10))
    hierarchy.dendrogram(
        corr_linkage, labels=features, leaf_rotation=90
    )
    plt.show()
    
    from collections import defaultdict
    
    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    return selected_features

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

def analyze_importances(clf, X_test, y_test, features):
    import eli5
    from eli5.sklearn import PermutationImportance

    permuter = PermutationImportance(
            estimator = clf, 
            scoring = 'accuracy', 
            n_iter = 1000,
            random_state=42).fit(X_test, y_test)
    feature_importance = permuter.feature_importances_
    #print(eli5.format_as_text(
        #eli5.explain_weights(clf, feature_names=features)))
    #print(eli5.format_as_text(
        #eli5.explain_weights(permuter, feature_names=features)))

    df = eli5.explain_weights_df(permuter, feature_names=features)
    print (df)
    df = df.sort_values('weight')

    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.barh(tree_indices, df['weight'].to_numpy(), height=0.7)
    ax.set_yticks(tree_indices)
    ax.set_yticklabels(df.feature.to_numpy())
    ax.set_ylim((0, df.shape[0]))
    plt.show()
    return df[(df['weight'] + df['std']) > 0.01].feature.to_list()

def find_permutation_importance(cdf, classes, features):
    ccdf = cdf.copy()
    train_index, test_index = get_full_indexes(cdf, classes, 0.8)
    ccdf_train = ccdf.loc[train_index]
    ccdf_test = ccdf.loc[test_index]
    X_train = ccdf_train.loc[:, features].values
    y_train = ccdf_train.loc[:, 'cluster.ID'].values
    X_test = ccdf_test.loc[:, features].values
    y_test = ccdf_test.loc[:, 'cluster.ID'].values

    n_classes = np.unique(y_train).shape[0]
    n_comp = min(len(features), n_classes-1)

    classifier = RandomForestClassifier(max_depth=n_comp, random_state=0)

    classifier.fit(X_train, y_train)
    return analyze_importances(classifier, X_test, y_test, features)

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

    features = np.array(['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5'])
    print ('FF:', list(features))
    print ('RF:', ['C_1', 'C_2', 'C_3', 'cos-theta_S-tau', 'lambda_3'])

    uncorr_features = find_selected_features(cdf, features)
    print ('UF:', list(features[uncorr_features]))

    classes = [1, 2, 4, 6, 7, 11, 12, 13, 14, 15, 16]
    important_features = find_permutation_importance(ccdf, classes, features[uncorr_features])
    print ('IF:', important_features)

def channel_flow():
    df = pd.read_csv('../../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../../toUFl.20210301/allChannel_features.dat', header=None, names=['y_plus', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'velocity_y', 'case.no'], delim_whitespace=True)

    ccdf = cdf.copy()
    ccdf['cluster.ID'] = df['cluster.ID']
    ccdf = ccdf.sort_values(by=['y_plus'])
    features = np.array(['C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5'])
    print ('FF:', list(features))
    print ('RF:', ['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3'])

    uncorr_features = find_selected_features(cdf, features)
    print ('UF:', list(features[uncorr_features]))

    classes = [1, 2, 3, 4, 5]
    important_features = find_permutation_importance(ccdf, classes, features[uncorr_features])
    print ('IF:', important_features)

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
    features = np.array(['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5'])
    print ('FF:', list(features))
    print ('RF:', ['C_1', 'C_2', 'C_3', 'eta_1'])

    uncorr_features = find_selected_features(cdf, features)
    print ('UF:', list(features[uncorr_features]))

    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    important_features = find_permutation_importance(ccdf, classes, features[uncorr_features])
    print ('IF:', important_features)

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
    features = np.array(['C_1', 'C_2', 'C_3', 'S-tau-1', 'S-tau-2', 'S-tau-3', 'S-omega-1', 'S-omega-2', 'tau-omega-1', 'tau-omega-2', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5'])
    print ('FF:', list(features))
    print ('RF:', ['C_1', 'C_2', 'C_3', 'cos theta_S-tau'])

    uncorr_features = find_selected_features(cdf, features)
    print ('UF:', list(features[uncorr_features]))

    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19]
    important_features = find_permutation_importance(ccdf, classes, features[uncorr_features])
    print ('IF:', important_features)

channel_flow()
#wavy_wall()
#channel_with_bump()
#square_cyl()

