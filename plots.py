import sys
import os
import numpy as np
sys.path.append('../')
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle as pkl
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn import feature_extraction
import gensim


def plot_doc_len_hist(lent, folder, dataset_name):
    log_bins = np.logspace(0, np.log10(max(lent)), base=10, num=50)
    plt.hist(lent, bins=log_bins)
    plt.xscale('log')
    plt.xlabel('Document length (words), logspace', fontdict={'size':15})
    plt.ylabel('Count', fontdict={'size': 15})
    plt.tight_layout()
    plt.savefig(os.path.join(folder, dataset_name + '.pdf'))
    plt.savefig(os.path.join(folder, dataset_name + '.png'))
    plt.clf()
    plt.close()

    # for Twitter (not log space)
    plt.hist(lent)
    plt.xlabel('Document length (words)', fontdict={'size':15})
    plt.ylabel('Count', fontdict={'size': 15})
    plt.tight_layout()
    plt.savefig(os.path.join(folder, dataset_name + '_2.pdf'))
    plt.savefig(os.path.join(folder, dataset_name + '_2.png'))
    plt.clf()
    plt.close()


def plot_history_independent(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]

        with open(run + '/model_history.pickle', 'rb') as handle:
            model_history = pkl.load(handle)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
        axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
        axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
        axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].legend(loc='lower right')

        axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
        axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
        axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
        axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Loss', fontsize=14)
        axes[1].legend(loc='lower right')

        # Save figure
        fig.savefig(plots_path + 'model_history' + model_name + '.png')
        fig.savefig(plots_path + 'model_history' + model_name + '.pdf')


def plot_history_with_2_outputs(main_folder, model_history):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
    axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
    axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
    axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(loc='lower right')

    axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
    axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
    axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
    axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(loc='lower right')

    # Save figure
    fig.savefig(main_folder + 'model_history.png')
    fig.savefig(main_folder + 'model_history.pdf')


def general_plot():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    axes[0].plot(x, y1, label=r'$f_1(x) = (1 + \frac{1}{x})^x$')
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel(r'$f_1(x) = (1 + \frac{1}{x})^x$', fontsize=10)
    axes[0].legend(loc='upper left')

    axes[1].plot(x, y2, label=r'$f_1(x) = (1 - \frac{1}{x})^x$')
    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel(r'$f_1(x) = (1 - \frac{1}{x})^x$', fontsize=10)
    axes[1].legend(loc='upper right')
    plt.show()

    plt.subplots(figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    y3 = y2*y1
    plt.plot(x, y1, label=r'$f(x) = (1 + \frac{1}{x})^x$')
    plt.plot(x, y2, label=r'$f(x) = (1 - \frac{1}{x})^x$')
    plt.plot(x, y3, label=r'$f(x) = (1 + \frac{1}{x})^x (1 - \frac{1}{x})^x$')
    plt.xlabel('x', fontsize=14)
    plt.ylabel(r'$f(x)$', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()


def plot_3_pca(data1, data2, data3, y_train, folder, plot_name='self'):
    # y = y_train[:, 1]
    y = y_train

    pca1_data1, pca2_data1 = make_pca(data1, n_components=2)
    pca1_data2, pca2_data2 = make_pca(data2, n_components=2)
    pca1_data3, pca2_data3 = make_pca(data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('PCA2', fontsize=8)
    axes[1].set_ylabel('PCA2', fontsize=8)
    axes[2].set_xlabel('PCA1', fontsize=8)
    axes[2].set_ylabel('PCA2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)

    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_pca_new(data1, data2, data3, y_train, folder, plot_name='self'):
    y = y_train[:, 1]
    pca1_data1, pca2_data1 = make_pca(data1, n_components=2)
    pca1_data2, pca2_data2 = make_pca(data2, n_components=2)
    pca1_data3, pca2_data3 = make_pca(data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[0].set_ylabel('PCA2', fontsize=14)
    axes[1].set_ylabel('PCA2', fontsize=14)
    axes[2].set_xlabel('PCA1', fontsize=14)
    axes[2].set_ylabel('PCA2', fontsize=14)
    axes[0].text(0.02, 0.03, 'VAE',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0].transAxes,
            color='black', fontsize=20)
    axes[1].text(0.02, 0.03, 'LDA',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1].transAxes,
            color='black', fontsize=20)
    axes[2].text(0.02, 0.03, 'VAE + LDA',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2].transAxes,
            color='black', fontsize=20)

    # axes[0].set_title('VAE features', fontsize=10)
    # axes[1].set_title('LDA features', fontsize=10)
    # axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_pca_new_trte(tr_data, te_data, y_train, y_test, folder, plot_name='self'):
    if len(y_train.shape) == 2:
        y = y_train[:, 1]
    else:
        y = y_train

    if len(y_test.shape) == 2:
        y_te = y_test[:, 1]
    else:
        y_te = y_test

    (tr_data1, tr_data2, tr_data3) = tr_data
    (te_data1, te_data2, te_data3) = te_data

    tr_pca1_data1, tr_pca2_data1 = make_pca(tr_data1, n_components=2)
    tr_pca1_data2, tr_pca2_data2 = make_pca(tr_data2, n_components=2)
    tr_pca1_data3, tr_pca2_data3 = make_pca(tr_data3, n_components=2)

    te_pca1_data1, te_pca2_data1 = make_pca(te_data1, n_components=2)
    te_pca1_data2, te_pca2_data2 = make_pca(te_data2, n_components=2)
    te_pca1_data3, te_pca2_data3 = make_pca(te_data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 0)[0])], tr_pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 1)[0])], tr_pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 0)[0])], tr_pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 1)[0])], tr_pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 0)[0])], tr_pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 1)[0])], tr_pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 0)[0])], te_pca2_data1[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 1)[0])], te_pca2_data1[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 0)[0])], te_pca2_data2[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 1)[0])], te_pca2_data2[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 0)[0])], te_pca2_data3[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 1)[0])], te_pca2_data3[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 0].set_ylabel('PCA2', fontsize=14)
    axes[1, 0].set_ylabel('PCA2', fontsize=14)
    axes[2, 0].set_ylabel('PCA2', fontsize=14)
    axes[2, 0].set_xlabel('PCA1', fontsize=14)
    axes[2, 1].set_xlabel('PCA1', fontsize=14)

    axes[0,0].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 0].transAxes,
            color='black', fontsize=20)
    axes[1,0].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 0].transAxes,
            color='black', fontsize=20)
    axes[2,0].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 0].transAxes,
            color='black', fontsize=20)

    axes[0,1].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 1].transAxes,
            color='black', fontsize=20)
    axes[1,1].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 1].transAxes,
            color='black', fontsize=20)
    axes[2,1].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 1].transAxes,
            color='black', fontsize=20)

    # axes[0].set_title('VAE features', fontsize=10)
    # axes[1].set_title('LDA features', fontsize=10)
    # axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_tsne_new_trte(tr_data, te_data, y_train, y_test, folder, plot_name='self'):

    if len(y_train.shape) == 2:
        y = y_train[:, 1]
    else:
        y = y_train

    if len(y_test.shape) == 2:
        y_te = y_test[:, 1]
    else:
        y_te = y_test

    (tr_data1, tr_data2, tr_data3) = tr_data
    (te_data1, te_data2, te_data3) = te_data

    tr_pca1_data1, tr_pca2_data1 = make_tSNE(tr_data1, tsne_perplexity=40, n_components=2)
    tr_pca1_data2, tr_pca2_data2 = make_tSNE(tr_data2, tsne_perplexity=40, n_components=2)
    tr_pca1_data3, tr_pca2_data3 = make_tSNE(tr_data3, tsne_perplexity=40, n_components=2)

    te_pca1_data1, te_pca2_data1 = make_tSNE(te_data1, tsne_perplexity=40, n_components=2)
    te_pca1_data2, te_pca2_data2 = make_tSNE(te_data2, tsne_perplexity=40, n_components=2)
    te_pca1_data3, te_pca2_data3 = make_tSNE(te_data3, tsne_perplexity=40, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 0)[0])], tr_pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 1)[0])], tr_pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 0)[0])], tr_pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 1)[0])], tr_pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 0)[0])], tr_pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 1)[0])], tr_pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 0)[0])], te_pca2_data1[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 1)[0])], te_pca2_data1[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 0)[0])], te_pca2_data2[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 1)[0])], te_pca2_data2[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 0)[0])], te_pca2_data3[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 1)[0])], te_pca2_data3[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 0].set_ylabel('tSNE2', fontsize=14)
    axes[1, 0].set_ylabel('tSNE2', fontsize=14)
    axes[2, 0].set_ylabel('tSNE2', fontsize=14)
    axes[2, 0].set_xlabel('tSNE1', fontsize=14)
    axes[2, 1].set_xlabel('tSNE1', fontsize=14)

    axes[0,0].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 0].transAxes,
            color='black', fontsize=20)
    axes[1,0].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 0].transAxes,
            color='black', fontsize=20)
    axes[2,0].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 0].transAxes,
            color='black', fontsize=20)

    axes[0,1].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 1].transAxes,
            color='black', fontsize=20)
    axes[1,1].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 1].transAxes,
            color='black', fontsize=20)
    axes[2,1].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 1].transAxes,
            color='black', fontsize=20)

    plt.tight_layout()
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.png')
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.pdf')


def plot_3_tsne(data1, data2, data3, y_train, folder, plot_name='self'):
    # y = y_train[:, 1]
    y = y_train
    pca1_data1, pca2_data1 = make_tSNE(data1, tsne_perplexity=40, n_components=2)
    pca1_data2, pca2_data2 = make_tSNE(data2, tsne_perplexity=40, n_components=2)
    pca1_data3, pca2_data3 = make_tSNE(data3, tsne_perplexity=40, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('tSNE2', fontsize=8)
    axes[1].set_ylabel('tSNE2', fontsize=8)
    axes[2].set_xlabel('tSNE1', fontsize=8)
    axes[2].set_ylabel('tSNE2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.png')
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.pdf')


def plot_pca_tsne(run_info, over_sampled=False):

    main_folder = run_info['main_folder']
    dataset_name = run_info['dataset_name']
    latent_dim = run_info['latent_dim']
    word2vec_dim = run_info['word2vec_dim']
    n_topics = run_info['n_topics']

    features_path = main_folder + 'features/'
    if over_sampled:
        vae_features_train = np.load(features_path + 'vae_ext_train.npy')
        vae_features_test = np.load(features_path + 'vae_ext_test.npy')
        y_train = np.load(main_folder + 'train_ext_label.npy')
        y_test = np.load(main_folder + 'test_ext_label.npy')
    else:
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        y_train = np.load(main_folder + 'train_label.npy')#[:, 1]
        y_test = np.load(main_folder + 'test_label.npy')#[:, 1]

    lda_features_train = np.load(features_path + 'lda_tfidf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    plot_name_tr = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_train'
    if over_sampled:
        plot_name_tr = plot_name_tr + '_ext'

    plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)

    plot_3_pca(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)

    plot_name_te = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_test'
    if over_sampled:
        plot_name_te = plot_name_te + '_ext'
    plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)
    plot_3_pca(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)

    tr_data = (vae_features_train, lda_features_train, data3_train)
    te_data = (vae_features_test, lda_features_test, data3_test)
    plot_3_pca_new_trte(tr_data, te_data, y_train, y_test, main_folder, plot_name=dataset_name + '_paper_PCA')
    plot_3_tsne_new_trte(tr_data, te_data, y_train, y_test, main_folder, plot_name=dataset_name + '_paper_tSNE')


def plot_accuracy_metrics(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier", style="Dataset", aspect=.45)
        leg = sns_plot._legend
        leg.set_bbox_to_anchor([0.99, 0.35])
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def FNR_FPR_table_latex(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if
            '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_df = pd.read_csv(run)
        model_df_filtered2 = model_df[model_df['Metric'] != 'Accuracy']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'Precision']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'Recall']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'F-score']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Classifier'] != 'SVM_nonlinear']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered2.loc[model_df_filtered2['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'
        model_df_filtered2 = model_df_filtered2.reset_index(drop=True)
        new_df = pd.DataFrame(data=0, columns=['Classifier', 'Metric', 'VAE_Train', 'VAE_Test', 'LDA_Train', 'LDA_Test',
                                               'VAE + LDA_Train', 'VAE + LDA_Test'], index=range(12))
        # classifiers = sorted(list(set(model_df_filtered2['Classifier'])))
        classifiers = ['SVM (linear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP', 'KNN (K = 3)']
        metrics = list(set(model_df_filtered2['Metric']))
        c = 0
        for i in range(len(classifiers)):
            for j in range(len(metrics)):
                new_df.loc[c, 'Classifier'] = classifiers[i]
                new_df.loc[c, 'Metric'] = metrics[j]
                c += 1

        for i in range(len(model_df_filtered2)):
            classifier = model_df_filtered2.loc[i, 'Classifier']
            value = model_df_filtered2.loc[i, 'Value']
            feature = model_df_filtered2.loc[i, 'Feature']
            dataset = model_df_filtered2.loc[i, 'Dataset']
            metric = model_df_filtered2.loc[i, 'Metric']
            new_df.loc[(new_df['Classifier'] == classifier) & (new_df['Metric'] == metric), feature + '_' + dataset] = round(value, 4)

        print(new_df.to_latex(index=False))


def plot_classifiers_result(run_info, over_sampled=False):
    main_folder = run_info['main_folder']
    if over_sampled:
        model_df = pd.read_csv(main_folder + 'Classifiers_over_sampled.csv')
        plot_name = 'Metrics_classifiers_over_sampled'
    else:
        model_df = pd.read_csv(main_folder + 'Classifiers.csv')
        plot_name = 'Metrics_classifiers'

    model_df_filtered = model_df[model_df['Metric'] != 'FNR']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
    model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier",
                           style="Dataset", aspect=.45)
    leg = sns_plot._legend
    leg.set_bbox_to_anchor([1.25, 0.4])
    plt.tight_layout()
    sns_plot.savefig(main_folder + plot_name + '.png')
    sns_plot.savefig(main_folder + plot_name + '.pdf')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()


def plot_accuracy_metrics2(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df['Metric'] != 'FPR']
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Classifier", y="Value", col="Feature", hue="Metric", style="Dataset")
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_WordClouds(data_tr, data_te, data_2_true, data_2_false, main_folder):
    wordcloud0 = WordCloud(background_color='white').generate(' '.join(data_tr['text']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(data_te['text']))
    wordcloud2 = WordCloud(background_color='white').generate(' '.join(data_2_true['text']))
    wordcloud3 = WordCloud(background_color='white').generate(' '.join(data_2_false['text']))

    plt.figure(figsize=(8, 8))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')
    i = 2

    ax2 = plt.subplot(gs1[i])
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis('off')

    i = 3
    ax3 = plt.subplot(gs1[i])
    plt.imshow(wordcloud3, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(main_folder + '/WorldCloud.png')


def plot_WordClouds_twitter(data_df_all, main_folder):
    # data_df_all = filtered_data_df_all
    real_no = data_df_all[data_df_all['label'] == 'real']
    fake_no = data_df_all[data_df_all['label'] == 'fake']

    wordcloud0 = WordCloud(background_color='white').generate(' '.join(real_no['text']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(fake_no['text']))

    plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')

    plt.savefig(main_folder + '/WorldCloud_twitter.png')
    plt.savefig(main_folder + '/WorldCloud_twitter.pdf')


def plot_WordClouds_covid(data_df_all, main_folder):

    real_no = data_df_all[data_df_all['label'] == 'real']
    fake_no = data_df_all[data_df_all['label'] == 'fake']

    wordcloud0 = WordCloud(background_color='white').generate(' '.join(real_no['headlines']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(fake_no['headlines']))

    plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')

    plt.savefig(main_folder + '/WorldCloud_covid.png')
    plt.savefig(main_folder + '/WorldCloud_covid.pdf')


def make_acc_prec_fscore_lines(isot_file, covid_file, twitter_file, classifier_name, metric, decimal_format):

    block_df = isot_file[((isot_file["Classifier"]==classifier_name) & (isot_file["Metric"]==metric))]
    vae_tr = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_cov = covid_file[((covid_file["Classifier"]==classifier_name) & (covid_file["Metric"]==metric))]
    vae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_twi = twitter_file[((twitter_file["Classifier"]==classifier_name) & (twitter_file["Metric"]==metric))]
    vae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    acc_pre_line = '\\multicolumn{1}{|l|}{} & Acc. & '
    prec_pre_line = '\\multicolumn{1}{|l|}{} & Prec. & '
    fscore_pre_line = '\\multicolumn{1}{|l|}{} & F-Sc. & '

    txt = '\\multicolumn{1}{l}{' +vae_te+ '} &\\multicolumn{1}{l}{' + lda_te + '} &\\multicolumn{1}{l|}{' +lvae_te+ '} & ' \
          +vae_te_cov+ ' & ' +lda_te_cov + ' & ' +lvae_te_cov + ' & ' + vae_te_twi+ ' & ' +lda_te_twi+ ' & ' + lvae_te_twi+ ' \\\\'

    if metric == 'Accuracy':
        line = acc_pre_line + txt + '\n'

    elif metric == 'Precision':
        line = prec_pre_line + txt + '\n'

    elif metric == 'F-score':
        line = fscore_pre_line + txt + '\n'

    else:
        line = 'error'
    print(line)
    return line


def make_recall_lines(isot_file, covid_file, twitter_file, classifier_name, decimal_format):

    block_df = isot_file[((isot_file["Classifier"]==classifier_name) & (isot_file["Metric"]=='Recall'))]
    vae_tr = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_cov = covid_file[((covid_file["Classifier"]==classifier_name) & (covid_file["Metric"]=='Recall'))]
    vae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_twi = twitter_file[((twitter_file["Classifier"]==classifier_name) & (twitter_file["Metric"]=='Recall'))]
    vae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    if classifier_name == 'SVM_nonlinear' or classifier_name == 'SVM_linear':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{SVM}} & Rec. & '

    elif classifier_name == 'Logistic Regression':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{LR}} & Rec. & '

    elif classifier_name == 'Random Forest':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{RF}} & Rec. & '

    elif classifier_name == 'Naive Bayes':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{NB}} & Rec. & '

    elif classifier_name == 'MLP':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{MLP}} & Rec. & '

    elif classifier_name == 'KNN (K = 3)':
        rec_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{KNN}} & Rec. & '
    else:
        rec_pre_line = 'Error'

    txt =  vae_te + ' & ' + lda_te+ ' & ' +lvae_te+' & ' +vae_te_cov+ ' & '  + lda_te_cov+' & ' + lvae_te_cov +\
          ' & ' +vae_te_twi+ ' & ' +lda_te_twi+ ' & ' +lvae_te_twi+ ' \\\\'

    line = rec_pre_line + txt + '\n'

    print(line)
    return line


def make_fscore_lines(isot_file, covid_file, twitter_file, classifier_name, decimal_format):
    block_df = isot_file[((isot_file["Classifier"] == classifier_name) & (isot_file["Metric"] == 'F-score'))]
    vae_tr = str(
        format(block_df[(block_df['Feature'] == 'VAE') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te = str(
        format(block_df[(block_df['Feature'] == 'VAE') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr = str(
        format(block_df[(block_df['Feature'] == 'LDA') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te = str(
        format(block_df[(block_df['Feature'] == 'LDA') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr = str(
        format(block_df[(block_df['Feature'] == 'VAE + LDA') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te = str(
        format(block_df[(block_df['Feature'] == 'VAE + LDA') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_cov = covid_file[((covid_file["Classifier"] == classifier_name) & (covid_file["Metric"] == 'F-score'))]
    vae_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_cov = str(
        format(block_df_cov[(block_df_cov['Feature'] == 'VAE') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lda_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'LDA') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_cov = str(
        format(block_df_cov[(block_df_cov['Feature'] == 'LDA') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lvae_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[
            0], decimal_format))
    lvae_te_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
        decimal_format))

    block_df_twi = twitter_file[
        ((twitter_file["Classifier"] == classifier_name) & (twitter_file["Metric"] == 'F-score'))]
    vae_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_twi = str(
        format(block_df_twi[(block_df_twi['Feature'] == 'VAE') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lda_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'LDA') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_twi = str(
        format(block_df_twi[(block_df_twi['Feature'] == 'LDA') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lvae_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[
            0], decimal_format))
    lvae_te_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
        decimal_format))

    fscore_pre_line = '\multicolumn{1}{|l|}{} & F-Sc. &'

    txt =  vae_te +  ' & ' +  lda_te + ' & ' +  lvae_te +  ' & ' + vae_te_cov +  ' & ' +  lda_te_cov + ' & ' +  lvae_te_cov  + \
          ' & '  + vae_te_twi +  ' & ' + lda_te_twi + ' & ' + lvae_te_twi + ' \\\\'

    line = fscore_pre_line + txt + ' \\hline \n'

    print(line)
    return line


def make_fpr_lines(isot_file, covid_file, twitter_file, classifier_name, decimal_format):

    block_df = isot_file[((isot_file["Classifier"]==classifier_name) & (isot_file["Metric"]=='FPR'))]
    vae_tr = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te = str(format(block_df[(block_df['Feature'] == 'VAE')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te = str(format(block_df[(block_df['Feature'] == 'LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te = str(format(block_df[(block_df['Feature'] == 'VAE + LDA')&(block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_cov = covid_file[((covid_file["Classifier"]==classifier_name) & (covid_file["Metric"]=='FPR'))]
    vae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_cov = str(format(block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA')&(block_df_cov['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_twi = twitter_file[((twitter_file["Classifier"]==classifier_name) & (twitter_file["Metric"]=='FPR'))]
    vae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te_twi = str(format(block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA')&(block_df_twi['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    if classifier_name == 'SVM_nonlinear' or classifier_name == 'SVM_linear':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{SVM}} & FPR & '

    elif classifier_name == 'Logistic Regression':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{LR}} & FPR & '

    elif classifier_name == 'Random Forest':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{RF}} & FPR & '

    elif classifier_name == 'Naive Bayes':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{NB}} & FPR & '

    elif classifier_name == 'MLP':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{MLP}} & FPR & '

    elif classifier_name == 'KNN (K = 3)':
        fpr_pre_line = '\\multicolumn{1}{|l|}{\\multirow{-2}{*}{KNN}} & FPR & '
    else:
        fpr_pre_line = 'Error'

    txt = vae_te+' & ' + lda_te+ ' & ' +lvae_te+ ' & ' + vae_te_cov+' & ' + lda_te_cov+' & ' + lvae_te_cov+ \
          ' & ' + vae_te_twi + ' & ' + lda_te_twi+ ' & ' + lvae_te_twi + ' \\\\'

    line = fpr_pre_line + txt + '\n'

    print(line)
    return line


def make_fnr_lines(isot_file, covid_file, twitter_file, classifier_name, decimal_format):
    block_df = isot_file[((isot_file["Classifier"] == classifier_name) & (isot_file["Metric"] == 'FNR'))]
    vae_tr = str(
        format(block_df[(block_df['Feature'] == 'VAE') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te = str(
        format(block_df[(block_df['Feature'] == 'VAE') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lda_tr = str(
        format(block_df[(block_df['Feature'] == 'LDA') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te = str(
        format(block_df[(block_df['Feature'] == 'LDA') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))
    lvae_tr = str(
        format(block_df[(block_df['Feature'] == 'VAE + LDA') & (block_df['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lvae_te = str(
        format(block_df[(block_df['Feature'] == 'VAE + LDA') & (block_df['Dataset'] == 'Test')]['Value'].values[0], decimal_format))

    block_df_cov = covid_file[((covid_file["Classifier"] == classifier_name) & (covid_file["Metric"] == 'FNR'))]
    vae_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_cov = str(
        format(block_df_cov[(block_df_cov['Feature'] == 'VAE') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lda_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'LDA') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_cov = str(
        format(block_df_cov[(block_df_cov['Feature'] == 'LDA') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lvae_tr_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA') & (block_df_cov['Dataset'] == 'Train')]['Value'].values[
            0], decimal_format))
    lvae_te_cov = str(format(
        block_df_cov[(block_df_cov['Feature'] == 'VAE + LDA') & (block_df_cov['Dataset'] == 'Test')]['Value'].values[0],
        decimal_format))

    block_df_twi = twitter_file[
        ((twitter_file["Classifier"] == classifier_name) & (twitter_file["Metric"] == 'FNR'))]
    vae_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    vae_te_twi = str(
        format(block_df_twi[(block_df_twi['Feature'] == 'VAE') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lda_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'LDA') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[0], decimal_format))
    lda_te_twi = str(
        format(block_df_twi[(block_df_twi['Feature'] == 'LDA') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
              decimal_format))
    lvae_tr_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA') & (block_df_twi['Dataset'] == 'Train')]['Value'].values[
            0], decimal_format))
    lvae_te_twi = str(format(
        block_df_twi[(block_df_twi['Feature'] == 'VAE + LDA') & (block_df_twi['Dataset'] == 'Test')]['Value'].values[0],
        decimal_format))

    fnr_pre_line = '\multicolumn{1}{|l|}{} & FNR &'

    txt =  vae_te + ' & ' +  lda_te + ' & ' +  lvae_te + ' & '  + vae_te_cov + ' & ' + lda_te_cov + ' & ' +  lvae_te_cov + \
          ' & ' + vae_te_twi + ' & ' + lda_te_twi + ' & ' +  lvae_te_twi + ' \\\\'

    line = fnr_pre_line + txt + ' \\hline \n'

    print(line)
    return line


def make_table_header():
    header =    '\\begin{table*}[h] \n' + \
                '\\caption{Blabla} \n' + \
                '\\label{tab:metrics_all} \n' + \
                '\\begin{tabular}{ll|lll|lll|lll|} \n' + \
                '\\cline{3-11} \n' + \
                '&  & \\multicolumn{3}{c|}{ISOT} & \\multicolumn{3}{c|}{Covid} & \\multicolumn{3}{c|}{Twitter} \\\\ \\cline{3-11} \n' + \
                ' & {\\color[HTML]{000000} } & \\multicolumn{1}{c|}{{\\color[HTML]{000000} VAE}} & \\multicolumn{1}{c|}{{\\color[HTML]{000000} LDA}} & \\multicolumn{1}{c|}{{\\color[HTML]{000000} VAE + LDA}} & \\multicolumn{1}{c|}{VAE} & \\multicolumn{1}{c|}{LDA} & \\multicolumn{1}{c|}{VAE + LDA} & \\multicolumn{1}{c|}{VAE} & \\multicolumn{1}{c|}{LDA} & \\multicolumn{1}{c|}{VAE + LDA} \\\\ \\hline \n'
    # + \
                # '\\multicolumn{1}{|l|}{CLF} & Metr. & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} & \\multicolumn{1}{c|}{Test} \\\\ \\hline \n'
    print(header)
    return header


def make_table_footer():
    footer = '\\end{tabular} \n' + '\\end{table*} \n'
    print(footer)
    return footer


def make_table(isot_file_path, covid_file_path, twitter_file_path, r=2):

    decimal_format = '.' + str(r) + 'f'

    isot_file = pd.read_csv(isot_file_path)
    covid_file = pd.read_csv(covid_file_path)
    twitter_file = pd.read_csv(twitter_file_path)

    classifiers = ['SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP', 'KNN (K = 3)']

    file1 = open("table.txt", "w")
    file1.write(make_table_header())
    for classifier in classifiers:
        file1.write(make_acc_prec_fscore_lines(isot_file, covid_file, twitter_file, classifier, 'Accuracy', decimal_format))
        file1.write(make_acc_prec_fscore_lines(isot_file, covid_file, twitter_file, classifier, 'F-score', decimal_format))
        file1.write(make_fpr_lines(isot_file, covid_file, twitter_file, classifier, decimal_format))
        file1.write(make_fnr_lines(isot_file, covid_file, twitter_file, classifier, decimal_format))

    file1.write(make_table_footer())

    file1.close()


def radar_factory(num_vars, frame='circle'):
    """
    credit to matplotlib.org
    Class copied from:
    https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def make_covid_data(k, top_folder, main_folder, run_info):

    # run_info_path = os.path.join(main_folder, 'run_info.pickle')
    # with open(run_info_path, 'rb') as handle:
    #     run_info = pkl.load(handle)
    #
    # word2vec_dim = run_info['word2vec_dim']
    # target_column = run_info['target_column']
    # dataset_name = run_info['dataset_name']
    run_folder = run_info['main_folder']
    word2vec_dim = 32
    target_column = 'text'
    dataset_name = 'COVID'

    my_stopwords = list(feature_extraction.text.ENGLISH_STOP_WORDS)

    this_data_folder = os.path.join(top_folder, 'data', dataset_name)

    with open(this_data_folder + '/word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index_orig = pkl.load(handle)

    wc = 0
    word_index = {}
    for key in word_index_orig:
        if key not in my_stopwords and len(key) > 2 and key != "n't":
            word_index[key] = wc
            wc += 1

    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + '/x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + '/x_test_' + str(word2vec_dim) + '.csv')

    data_df_all = x_train.append(x_test, ignore_index=True)

    real_df = data_df_all[data_df_all['label'] == 'real'].reset_index(drop=True)
    fake_df = data_df_all[data_df_all['label'] == 'fake'].reset_index(drop=True)
    #
    # # if gensim model for lda is used:
    # temp_file = os.path.join(main_folder, "Covid Lda/lda_model_" + str(k))
    # lda = gensim.models.ldamodel.LdaModel.load(temp_file)
    #
    # real_documents = list(real_df[target_column])
    # real_texts = [[word for word in document.split()] for document in real_documents]
    # real_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
    #                real_texts]
    #
    # real_gensim = np.array(lda.get_document_topics(real_corpus))
    # lda_theta_real = np.zeros([len(real_gensim), k])
    #
    # for i in range(len(real_gensim)):
    #     for elem in real_gensim[i]:
    #         lda_theta_real[i, elem[0]] = elem[1]
    #
    # fake_documents = list(fake_df[target_column])
    # fake_texts = [[word for word in document.split()] for document in fake_documents]
    # fake_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
    #                fake_texts]
    #
    # fake_gensim = np.array(lda.get_document_topics(fake_corpus))
    # lda_theta_fake = np.zeros([len(fake_gensim), k])
    #
    # for i in range(len(fake_gensim)):
    #     for elem in fake_gensim[i]:
    #         lda_theta_fake[i, elem[0]] = elem[1]
    #
    # cov_data = ('Covid', [
    #     list(np.mean(lda_theta_real, axis=0)),
    #     list(np.mean(lda_theta_fake, axis=0))])

    # if sklearn model of LDA is used:
    x_train_lda_features = np.load(run_folder + 'features/lda_tfidf_train.npy')
    x_test_lda_features = np.load(run_folder + 'features/lda_tfidf_test.npy')
    lda_features = np.concatenate([x_train_lda_features, x_test_lda_features], axis=0)
    
    real_idx = data_df_all.index[data_df_all['label'] == 'real'].tolist()
    fake_idx = data_df_all.index[data_df_all['label'] == 'fake'].tolist()
    
    lda_theta_real = lda_features[real_idx, :]
    lda_theta_fake = lda_features[fake_idx, :]
    #
    # topic0_inclusion_sorted = np.argsort(lda_theta_fake[:, 0])[::-1]
    # interestingwords = ['government', 'say', 'spread', 'vaccine', 'claims', 'positive', 'food', 'times',
    #                     'wuhan', 'allegedly', 'test' ,'photo', 'virus', 'facebook', 'novel']
    #
    # for id in range(len(topic0_inclusion_sorted)):
    #     idx=topic0_inclusion_sorted[id]
    #     score = len(set(fake_df.loc[idx, 'text'].split(' ')).intersection(interestingwords))
    #     # if 'facebook' in fake_df.loc[idx, 'text'] and 'photo' in fake_df.loc[idx, 'text'] and 'claim' in fake_df.loc[idx, 'text'] and 'spread' in fake_df.loc[idx, 'text']:
    #     if score > 3:
    #         print(id, idx, fake_df.loc[idx, 'text'])
    #
    # # lda_theta_fake[6697, :]
    # lda_theta_fake[6813, :]
    # lda_theta_fake[7357, :]
    #
    # topic3_inclusion_sorted = np.argsort(lda_theta_real[:,3])[::-1]
    # interestingwords_real = ['need', 'virus', 'home', 'mask', 'hospital', 'infected', 'emergency', 'stay', 'brazil', 'patient', 'president', 'going',
    #                     'staff', 'tests', 'plan' ,'body', 'medical', 'province', 'president', 'thousands', 'italy', 'spanish',
    #                          'states', 'information', 'community', 'emergency','news']
    #
    # for id in range(len(topic3_inclusion_sorted)):
    #     idx=topic3_inclusion_sorted[id]
    #     score = len(set(real_df.loc[idx, 'text'].split(' ')).intersection(set(interestingwords_real)))
    #     if score > 3:
    #         print(id, idx, real_df.loc[idx, 'text'])
    #
    # lda_theta_real[2274, :]
    # lda_theta_real[1370, :]
    # lda_theta_real[1588, :]
    #
    # lda_theta_real[4381, :]
    #
    # lda_theta_real[426, :]
    #
    #
    # lda_theta_real[5088, :]
    #
    cov_data = ('COVID', [
        list(np.mean(lda_theta_real, axis=0)),
        list(np.mean(lda_theta_fake, axis=0))])
    return cov_data


def make_twitter_data(k, top_folder, main_folder):
    # run_info_path = os.path.join(main_folder, 'run_info.pickle')
    # with open(run_info_path, 'rb') as handle:
    #     run_info = pkl.load(handle)
    #
    # word2vec_dim = run_info['word2vec_dim']
    # target_column = run_info['target_column']
    # dataset_name = run_info['dataset_name']

    word2vec_dim = 32
    target_column = 'text'
    dataset_name = 'Twitter'

    my_stopwords = list(feature_extraction.text.ENGLISH_STOP_WORDS)

    this_data_folder = os.path.join(top_folder, 'data', dataset_name)


    with open(this_data_folder + '/word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index_orig = pkl.load(handle)

    wc = 0
    word_index = {}
    index_2_word = {}
    for key in word_index_orig:
        if key not in my_stopwords and len(key) > 2 and key != "n't":
            word_index[key] = wc
            index_2_word[wc]=key
            wc += 1

    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + '/x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + '/x_test_' + str(word2vec_dim) + '.csv')

    data_df_all = x_train.append(x_test, ignore_index=True)

    temp_file = os.path.join(main_folder, "Twitter Lda/lda_model_" + str(k))
    lda = gensim.models.ldamodel.LdaModel.load(temp_file)

    real_df = data_df_all[data_df_all['label'] == 'real']
    fake_df = data_df_all[data_df_all['label'] == 'fake']

    real_documents = list(real_df[target_column])
    real_texts = [[word for word in document.split()] for document in real_documents]
    real_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
                   real_texts]

    real_gensim = np.array(lda.get_document_topics(real_corpus))
    lda_theta_real = np.zeros([len(real_gensim), k])

    for i in range(len(real_gensim)):
        for elem in real_gensim[i]:
            lda_theta_real[i, elem[0]] = elem[1]

    fake_documents = list(fake_df[target_column])
    fake_texts = [[word for word in document.split()] for document in fake_documents]
    fake_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
                   fake_texts]

    fake_gensim = np.array(lda.get_document_topics(fake_corpus))
    lda_theta_fake = np.zeros([len(fake_gensim), k])

    for i in range(len(fake_gensim)):
        for elem in fake_gensim[i]:
            lda_theta_fake[i, elem[0]] = elem[1]

    t_data = ('Twitter', [
        list(np.mean(lda_theta_real, axis=0)),
        list(np.mean(lda_theta_fake, axis=0))])

    return t_data


def make_isot_data(k, top_folder, main_folder):
    # run_info_path = os.path.join(main_folder, 'run_info.pickle')
    # with open(run_info_path, 'rb') as handle:
    #     run_info = pkl.load(handle)

    # word2vec_dim = run_info['word2vec_dim']
    # target_column = run_info['target_column']
    # dataset_name = run_info['dataset_name']

    word2vec_dim = 32
    target_column = 'text'
    dataset_name = 'ISOT'

    my_stopwords = list(feature_extraction.text.ENGLISH_STOP_WORDS)

    this_data_folder = os.path.join(top_folder, 'data', dataset_name)

    with open(this_data_folder + '/word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index_orig = pkl.load(handle)

    wc = 0
    word_index = {}
    index_2_word = {}
    for key in word_index_orig:
        if key not in my_stopwords and len(key) > 2 and key != "n't":
            word_index[key] = wc
            index_2_word[wc] = key
            wc += 1
    
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + '/x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + '/x_test_' + str(word2vec_dim) + '.csv')

    data_df_all = x_train.append(x_test, ignore_index=True)

    temp_file = os.path.join(main_folder, "ISOT Lda/lda_model_" + str(k))
    lda = gensim.models.ldamodel.LdaModel.load(temp_file)

    real_df = data_df_all[data_df_all['label'] == 'real']
    fake_df = data_df_all[data_df_all['label'] == 'fake']

    real_documents = list(real_df[target_column])
    real_texts = [[word for word in document.split()] for document in real_documents]
    real_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
                   real_texts]

    real_gensim = np.array(lda.get_document_topics(real_corpus))
    lda_theta_real = np.zeros([len(real_gensim), k])

    for i in range(len(real_gensim)):
        for elem in real_gensim[i]:
            lda_theta_real[i, elem[0]] = elem[1]

    fake_documents = list(fake_df[target_column])
    fake_texts = [[word for word in document.split()] for document in fake_documents]
    fake_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in
                   fake_texts]

    # ss = lda.get_document_topics(fake_corpus)
    # a = np.array(fake_corpus)

    fake_gensim = np.array(lda.get_document_topics(fake_corpus))
    lda_theta_fake = np.zeros([len(fake_gensim), k])

    for i in range(len(fake_gensim)):
        for elem in fake_gensim[i]:
            lda_theta_fake[i, elem[0]] = elem[1]

    is_data = ('ISOT', [
        list(np.mean(lda_theta_real, axis=0)),
        list(np.mean(lda_theta_fake, axis=0))])

    return is_data


def make_radar_plot_data(top_folder, main_folder, categories, k):
    isot_data = make_isot_data(k, top_folder, main_folder)

    cov_data = make_covid_data(k, top_folder, main_folder)

    twitter_data = make_twitter_data(k, top_folder, main_folder)

    data = [categories, isot_data, cov_data, twitter_data]

    return data


def plot_subject_credibility():

    main_folder = 'LDA/'
    top_folder = 'runs/'
    for k in [5, 10, 32]:
        categories = ['Topic ' + str(i) for i in range(k)]

        data = make_radar_plot_data(top_folder, main_folder, categories, k)

        N = k
        theta = radar_factory(N, frame='polygon')

        spoke_labels = data.pop(0)

        fig, axs = plt.subplots(figsize=(19, 5), nrows=1, ncols=3,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        colors = ['b', 'r']

        for ax, (title, case_data) in zip(axs.flat, data):
            # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25)
            ax.set_varlabels(spoke_labels)

        # add legend relative to top-left plot
        labels = ('Real', 'Fake')
        legend = axs[2].legend(labels, loc=(0.9, .95),
                               labelspacing=0.1, fontsize='small')

        # fig.text(0.5, 0.965, 'Subject Credibility',
        #          horizontalalignment='center', color='black', weight='bold',
        #          size='large')

        plt.savefig(main_folder + 'radar_subjects_' + str(k) + '.png')
        plt.savefig(main_folder + 'radar_subjects_' + str(k) + '.pdf')


def plot_varphi():
    plt.cla
    plt.close()
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))

    x=[i/2 for i in range(10)]
    y=[0.77025309, 0.01468381, 0.01468356, 0.01468789, 0.11226562,
       0.01468376, 0.01468381, 0.01468685, 0.014684 , 0.0146876]
    
    plt.bar(x, height= y, width=0.3,color='white', edgecolor='black')
    xlocs, xlabs = plt.xticks()
    xlocs=[i/2 for i in range(0,10)]
    xlabs=[r'$\varphi_{i' + str(i)+ '}$' for i in range(0,10)]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel("Topics", fontsize = 22)
    # plt.ylabel(r'$\varphi_{i.}$', fontsize = 25)
    plt.xticks(xlocs, xlabs)
    plt.tight_layout()
    fig.savefig('LDA/fake2.png')
    fig.savefig('LDA/fake2.pdf')


    plt.cla
    plt.close()
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))

    x = [i/2 for i in range(10)]
    # y = [0.02296503, 0.02297805, 0.02296167, 0.5448321 , 0.02296149,
    #    0.02296169, 0.27146385, 0.02295799, 0.022958 , 0.02296013]
    y = [0.02241635, 0.13368982, 0.02242124, 0.62071895, 0.02241683,
                0.02241661, 0.02241731, 0.08866812, 0.02241625, 0.02241852]
    plt.bar(x, height= y, width=0.3,color='white', edgecolor='black')
    xlocs, xlabs = plt.xticks()
    xlocs=[i/2 for i in range(0, 10)]
    xlabs=[r'$\varphi_{j' + str(i)+ '}$' for i in range(0,10)]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.xlabel("Topics", fontsize = 22)
    # plt.ylabel(r'$\varphi_{j.}$', fontsize = 25)
    plt.xticks(xlocs, xlabs)
    plt.tight_layout()
    fig.savefig('LDA/real2.png')
    fig.savefig('LDA/real2.pdf')


def plot_top_words(model, feature_names, n_top_words, title, top_folder, dataset_name, word2vec_dim, my_stopwords,
                   index_2_word):
    this_data_folder = os.path.join(top_folder, 'data', dataset_name)

    with open(this_data_folder + '/word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index_orig = pkl.load(handle)

    wc = 0
    word_index = {}
    index_2_word = {}
    for key in word_index_orig:
        if key not in my_stopwords and len(key) > 2 and key != "n't":
            word_index[key] = wc
            index_2_word[wc]=key
            wc += 1
    
    fig, axes = plt.subplots(1, 10, figsize=(30, 6), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        print(topic_idx)#, topic, len(topic))
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        for ii in range(len(top_features)):
            print(top_features[ii], index_2_word[int(top_features[ii])])
        
        ax = axes[topic_idx]
        ax.barh([index_2_word[int(top_features[i])] for i in range(len(top_features))], weights, height=0.7)
        # ax.set_title(f'Topic  {topic_idx +1}', fontdict={'fontsize': 15})
        id_str = str(topic_idx)
        eq = r'$(\beta_{k=' + id_str + '})$'
        txt = 'Topic ' + id_str + ' '
        # aa = r'$\text\{Topic {0}\} (\beta_\{k={1}\})$'.format(sss, sss)
        
        ax.set_title(txt + eq, fontdict={'fontsize': 15})
        
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=13)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        # fig.suptitle(title, fontsize=25)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.30, hspace=0.1)
    plt.savefig(title+'.png')
    plt.savefig(title+'.pdf')


def plot_top_words_topics_separately(model, feature_names, n_top_words, path, index_2_word):
    path = 'LDA/'
    selected_topics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    selected_comps = model.components_[selected_topics,:]
    # colors = ['red', 'blue', 'blue', 'blue', 'red', 'blue', 'red', 'blue', 'red', 'red']
    # colors = ['red', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'blue', 'blue', 'blue']
    colors = ['blue', 'blue', 'blue', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    
    for topic_idx, topic in enumerate(selected_comps):
        fig, axes = plt.subplots(1, 1, figsize=(7, 6), sharex=True)
        # axes = axes.flatten()
        normalized_topic = topic/np.sum(topic)
        print(topic_idx)#, topic, len(topic))
        top_features_ind = normalized_topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = normalized_topic[top_features_ind]
        
        # for ii in range(len(top_features)):
        #     print(top_features[ii], index_2_word[int(top_features[ii])])
        ax = axes
        # ax.barh([index_2_word[int(top_features[i])] for i in range(len(top_features))], weights, height=0.7, color=colors[topic_idx], alpha=0.5)
        ax.barh([index_2_word[int(top_features[i])] for i in range(len(top_features))], weights, height=0.7, color='red', edgecolor='black')
        plt.xticks(rotation= 90)
        # ax.set_title(f'Topic  {topic_idx +1}', fontdict={'fontsize': 15})
        id_str = str(selected_topics[topic_idx])
        
        eq = r'$(\beta_{k=' + id_str + '})$'
        txt = 'Topic ' + id_str + ' '
        # aa = r'$\text\{Topic {0}\} (\beta_\{k={1}\})$'.format(sss, sss)
        
        ax.set_title(txt + eq, fontdict={'fontsize': 15})
        
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=13)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        # fig.suptitle(title, fontsize=25)
        
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.30, hspace=0.3)
        plt.tight_layout()
        plt.savefig(path+'covid_topic_new' + str(topic_idx) + '.png')
        plt.savefig(path+'covid_topic_new' + str(topic_idx) + '.pdf')


def find_latent_assignments(model, feature_names, n_top_words, path, word_index):
    path = 'LDA/'
    sentence = 'a post shared more than a thousand times on facebook claims that a corpse of a covid 19 positive person ' \
               'is 100 times more aeuroetoxicaeur 72 hours after death aeur and that because undertakers are not ' \
               'burying bodies within this prescribed period funerals have become hotspots for further infections'
    sentence1 = 'stay home stay informed ontario reports 1st Covid 19 related death as province declares state of ' \
                'emergency'
    sentence_words = sentence1.split(' ')
    for word in sentence_words:
        if word in word_index:
            word_id = word_index[word]
            if str(word_id) in feature_names:
                beta_idx = feature_names.index(str(word_id))
                topic_id = np.argmax(model.components_[:, beta_idx])
                print(word, topic_id)#, model.components_[:, beta_idx])
