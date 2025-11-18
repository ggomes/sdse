import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

from matplotlib.colors import LinearSegmentedColormap, to_rgb

c0 = '#e58139'
c1 = '#399de5'

# c0 = 'r'
# c1 = 'g'

def generate_mesh(X):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
    return xx, yy


def disp_tree(treeX,figsize,fontsize,feature_names):
    plt.figure(figsize=figsize)
    tree.plot_tree(treeX,
                   fontsize=fontsize,
                   feature_names=feature_names,
                   label='all',
                   filled=True,
                   rounded=True)


def disp_tree_classnames(treeX,figsize,fontsize,feature_names,class_names):
    plt.figure(figsize=figsize)
    tree.plot_tree(treeX,
                   fontsize=fontsize,
                   feature_names=feature_names,
                   label='all',
                   filled=True,
                   rounded=True,
                   class_names=class_names)


def plot_tree(treeX,xx,yy,X0,X1,figsize=(10,10)):

    fig, ax = plt.subplots(figsize=figsize)

    Z = treeX.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.3,levels=[0.1,0.9], colors=[c0,c1], extend='both')
    cs.cmap.set_over(c1)
    cs.cmap.set_under(c0)
    cs.changed()

    plt.scatter(X0[:,0],X0[:,1],marker='o',s=60, linewidths=2,facecolors='none', edgecolors=c0)
    plt.scatter(X1[:,0],X1[:,1],marker='x',s=60,c=c1, linewidths=2)

    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    ax.spines[:].set_visible(False)

def plot_tree_proba(treeX, xx, yy, X0, X1, nbins,figsize=(10,10)):

    cmap = LinearSegmentedColormap.from_list('sadf', [to_rgb(c0), to_rgb(c1)], N=nbins)

    fig, ax = plt.subplots(figsize=figsize)

    Z = treeX.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, extend='both')

    cs.cmap.set_over(c1)
    cs.cmap.set_under(c0)
    cs.changed()

    plt.scatter(X0[:, 0], X0[:, 1], marker='o', s=60, linewidths=2, facecolors='none', edgecolors=c0)
    plt.scatter(X1[:, 0], X1[:, 1], marker='x', s=60, c=c1, linewidths=2)

    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    ax.spines[:].set_visible(False)




def generate_spiral(N=400):

    theta = np.sqrt(np.random.rand(N)) * 2 * np.pi  # np.linspace(0,2*pi,100)

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + 2*np.random.randn(N, 2)

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + 2*np.random.randn(N, 2)

    res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
    res_b = np.append(x_b, np.ones((N, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    X = res[:, :2]
    y = res[:, 2]

    return X, y





