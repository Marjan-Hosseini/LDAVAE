from LDAVAE.vae import *
from LDAVAE.lda import *
from preprocessing import *
from plots import *
from classifiers import *
sys.path.append('../')


def n_topic_cross_validation(corpus, tr_corpus, te_corpus, d, main_folder):
    coherence_u_mass_all = []
    coherence_u_mass_tr = []
    coherence_u_mass_test = []

    if not os.path.exists(os.path.join(main_folder, 'ldamodels')):
        os.mkdir(os.path.join(main_folder, 'ldamodels'))

    for ntop in [2, 5, 10, 32, 50, 64, 100]:
        print(ntop)
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(corpus, num_topics=ntop, id2word=d, passes=10, iterations=500, chunksize=10000, eval_every=1)
        temp_file = os.path.join(main_folder, 'ldamodels/', "lda_model_" + str(ntop))

        with open(os.path.join(main_folder, 'ldamodels/', "lda_model_" + str(ntop) + '_topic_words.txt'), 'w') as f:
            for top in range(ntop):
                topw = ldamodel.get_topic_terms(topicid=top, topn=10)
                topwords = [(d[i[0]], str(i[1])) for i in topw]
                f.write("%s\n" % topwords)

        ldamodel.save(temp_file)
        # ldamodel = gensim.models.ldamodel.LdaModel.load(main_folder + "lda_model_" + str(k))
        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=corpus, dictionary=d,
                                                         coherence='u_mass')
        coherence_u_mass_all.append((ntop, cm.get_coherence()))

        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=tr_corpus, dictionary=d,
                                                         coherence='u_mass')
        coherence_u_mass_tr.append((ntop, cm.get_coherence()))

        cm2 = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=te_corpus, dictionary=d,
                                                          coherence='u_mass')
        coherence_u_mass_test.append((ntop, cm2.get_coherence()))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_all.txt'), 'w') as f:
        for elem in coherence_u_mass_all:
            f.write("%s\n" % str(elem))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_tr.txt'), 'w') as f:
        for elem in coherence_u_mass_tr:
            f.write("%s\n" % str(elem))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_te.txt'), 'w') as f:
        for elem in coherence_u_mass_test:
            f.write("%s\n" % str(elem))

    plt.clf()
    plt.plot([ee[0] for ee in coherence_u_mass_tr], [ee[1] for ee in coherence_u_mass_tr], 'x-', label='Train')
    plt.plot([ee[0] for ee in coherence_u_mass_test], [ee[1] for ee in coherence_u_mass_test], 'o-', label='Test')
    # plt.plot([ee[0] for ee in coherence_u_mass_all], [ee[1] for ee in coherence_u_mass_all], 'o-', label='Both')

    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.xticks([ee[0] for ee in coherence_u_mass_tr])
    plt.grid('on')
    plt.legend()
    plt.savefig(main_folder + 'lda_cross_validation.png')
    plt.savefig(main_folder + 'lda_cross_validation.pdf')


def concat_features(run_info):

    main_folder = run_info['main_folder']
    included_features = run_info['included_features']
    print('Concatenating feature sets in: ', included_features)

    # included_features = ['vae', 'lda_tf', 'lda_tfidf']
    address_tr = main_folder + 'features/' + included_features[0] + '_train.npy'
    features_tr = np.load(address_tr)

    address_te = main_folder + 'features/' + included_features[0] + '_test.npy'
    features_te = np.load(address_te)
    feat_name = included_features[0]
    if len(included_features) > 1:

        for feat in included_features[1:]:
            address_tr = main_folder + 'features/' + feat + '_train.npy'
            feat_tr = np.load(address_tr)
            features_tr = np.concatenate((features_tr, feat_tr), axis=1)

            address_te = main_folder + 'features/' + feat + '_test.npy'
            feat_te = np.load(address_te)
            features_te = np.concatenate((features_te, feat_te), axis=1)
            feat_name = feat_name + '_' + feat
    # vae_features_tr = np.load(main_folder + 'features/vae_train.npy')
    # vae_features_te = np.load(main_folder + 'features/vae_test.npy')
    # lda_features_tf_tr = np.load(main_folder + 'features/lda_tf_train.npy')
    # lda_features_tf_te = np.load(main_folder + 'features/lda_tf_test.npy')
    # lda_features_tfidf_tr = np.load(main_folder + 'features/lda_tfidf_train.npy')
    # lda_features_tfidf_te = np.load(main_folder + 'features/lda_tfidf_test.npy')
    #
    # lvae_features_tr = np.concatenate((vae_features_tr, lda_features_tf_tr), axis=1)
    # lvae_features_te = np.concatenate((vae_features_te, lda_features_tf_te), axis=1)

    # save the features
    np.save(main_folder + 'features/lvae_train_' + feat_name, features_tr)
    np.save(main_folder + 'features/lvae_test_' + feat_name, features_te)


def make_results_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
            # and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']

        data_tr = [vae_features_train, lda_features_train, data3_train]
        data_te = [vae_features_test, lda_features_test, data3_test]
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
        ct = 0
        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
                results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
                results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
                results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
                results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
                results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
                results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
                results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
                results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
                results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
                results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
                results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
                ct += 12
            results_pd.to_csv(results_path + model_name + '.csv', index=False)


def make_classifiers_df(run_info, k, over_sampled=False):
    print('Evaluating the features by classifiers ...')
    main_folder = run_info['main_folder']

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

    lda_features_train = np.load(features_path + 'lda_tfidf_train_' + str(k) + '.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test_' + str(k) + '.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']

    data_tr = [vae_features_train, lda_features_train, data3_train]
    data_te = [vae_features_test, lda_features_test, data3_test]
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
    ct = 0
    for c in range(len(classifiers)):
        classifier_name = classifiers[c]
        print(classifier_name)

        classifier = classifiers_funcs[c]
        # print(classifier_name)
        for d in range(len(data_tr)):
            print(d)
            this_tr = data_tr[d]
            this_te = data_te[d]
            this_feature = features[d]
            # print(this_feature)
            tr, te = classifier(this_tr, this_te, y_train, y_test)
            accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
            accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
            results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
            results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
            results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
            results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
            results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
            results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
            results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
            results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
            results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
            results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
            results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
            results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
            ct += 12
        # results_pd.to_csv(results_path + model_name + '.csv', index=False)
        if over_sampled:
            results_pd.to_csv(main_folder + 'Classifiers_over_sampled.csv', index=False)
        else:
            results_pd.to_csv(main_folder + 'Classifiers_lda_' + str(k) + '.csv', index=False)


def evaluation_expermient(info):
    print('Evaluating the results ...')
    # or you can fix on one k (topics)
    for k in [5, 10, 32, 50, 64, 100]:
        make_classifiers_df(info, k)
        print(k)
    # plot_classifiers_result(info)


def main(run_info):
    main_folder = run_info['main_folder']

    # data preparation
    if not os.path.exists(top_folder + 'data/' + dataset_name + '/x_train_' + str(word2vec_dim) + '.csv') or \
            not os.path.exists(top_folder + 'data/' + dataset_name + '/x_test_' + str(word2vec_dim) + '.csv'):
        prepare_data(run_info, top_folder, dataset_address)

    # VAE
    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/vae_test.npy'):
        vae_experiment(run_info, top_folder)

    # LDA
    if not os.path.exists(main_folder + 'features/lda_tfidf_train.npy') or \
            not os.path.exists(main_folder + 'features/lda_tfidf_test.npy'):
        lda_experiment(run_info, top_folder)

    # LDAVAE (VAE + LDA): Our method
    if not os.path.exists(main_folder + 'features/lvae_train_vae_lda_tfidf.npy') or \
            not os.path.exists(main_folder + 'features/lvae_test_vae_lda_tfidf.npy'):
        concat_features(run_info)

    # classification and evaluation
    evaluation_expermient(run_info)

    # run this function only if results for runs are not already computed
    lvae_classifier(run_info)


if __name__ == '__main__':

    if '-f' in sys.argv:
        top_folder = sys.argv[sys.argv.index('-f') + 1]
    else:
        top_folder = 'runs/'

    if '-d' in sys.argv:
        dataset_name = sys.argv[sys.argv.index('-d') + 1]
    else:
        dataset_name = exit('Error: You need to specify the dataset name with -d command. \nDatasets choices could be '
                            'Twitter or ISOT or Covid.')
        # dataset_name = 'ISOT'

    if '-a' in sys.argv:
        dataset_address = sys.argv[sys.argv.index('-a') + 1]
    else:
        dataset_address = exit('Error: You need to specify the address of top folder contatining both dataset folders '
                               'with -a command, eg. -a "data/".')
        # dataset_address = 'data/'

    if '-e' in sys.argv:
        epoch_no = int(sys.argv[sys.argv.index('-e') + 1])
    else:
        epoch_no = 1

    if '-t' in sys.argv:
        n_topics = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        n_topics = 10

    if '-i' in sys.argv:
        n_iter = int(sys.argv[sys.argv.index('-i') + 1])

    else:
        n_iter = 10

    if '-l' in sys.argv:
        latent_dim = int(sys.argv[sys.argv.index('-l') + 1])
    else:
        latent_dim = 32

    if '-w' in sys.argv:
        word2vec_dim = int(sys.argv[sys.argv.index('-w') + 1])
    else:
        word2vec_dim = 32

    run_info = make_run_info(top_folder, dataset_name, latent_dim, epoch_no, n_topics, n_iter, word2vec_dim)

    main(run_info)
