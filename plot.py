import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_sorted_cid_and_plot(df, fig, ax):
    n_points = df.shape[0]
    cluster_ids_unique = df['cluster.ID'].unique()
    n_clusters = len(cluster_ids_unique) ;
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
        rows = df[df['cluster.ID'] == cluster_id]
        ax.scatter(rows['x'], rows['y'], s=0.05, color=cols(cluster_id), label=str(cluster_id))
    plt.legend(loc='right', title='Clusters')
    plt.xlabel('x')
    plt.ylabel('y')
    return 

def channel_with_bump():
    df = pd.read_csv('../toUFl.20210222/channel.with.bump/data/bump_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'cos-theta_S-tau', 'lambda_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    
    
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)

    plt.show()

def channel_flow():
    df = pd.read_csv('../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    
    
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)
    plt.yscale('log')
    plt.show()

def wavy_wall():
    df = pd.read_csv('../toUFl.20210222/wavy.wall/data/wavywall_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    
    
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)
    #plt.yscale('log')
    plt.show()

def square_cyl():
    df = pd.read_csv('../toUFl.20210222/square.cylinder/data/sqcyl_clustering_11-4-2020.dat', header=0, names=['C_1', 'C_2', 'C_3', 'cos theta_S-tau', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    
    
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)
    plt.xscale('log')
    plt.show()

channel_flow()
wavy_wall()
channel_with_bump()
square_cyl()

