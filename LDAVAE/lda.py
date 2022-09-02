from preprocessing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

sys.path.append('../')
np.random.seed(2021)


def lda_experiment(run_info, top_folder):
    main_folder = run_info['main_folder']
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    n_topics = run_info['n_topics']
    n_top_words = run_info['n_top_words']
    n_features = run_info['n_features']
    n_iter = run_info['n_iter']
    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    print('LDA experiment with', n_topics, 'topics:')
    with open(main_folder + 'run_info.pickle', 'wb') as handle:
        pkl.dump(run_info, handle)
    
    this_data_folder = top_folder + 'data/' + dataset_name + '/'
    
    x_train = np.load(this_data_folder + 'train_text_' + str(word2vec_dim) + '.npy')
    x_test = np.load(this_data_folder + 'test_text_' + str(word2vec_dim) + '.npy')
    
    x_df_all = np.concatenate([x_train, x_test], axis=0)
    new_df = []
    for i in range(x_df_all.shape[0]):
        temp = []
        for j in range(x_df_all.shape[1]):
            temp.append(str(x_df_all[i, j]))
        xx = ' '.join(temp)
        new_df.append(xx)
    
    # new_df = list(data_df_all['headlines'])
    # new_df = list(data_df_all['title'])
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, lowercase=False, max_features=n_features,
                                       stop_words={'english'}, analyzer='word')
    tfidf = tfidf_vectorizer.fit_transform(new_df)

    tfidf_train = tfidf_vectorizer.transform(x_train)
    
    with open(main_folder + 'lda_vectorizer_tfidf.pkl', 'wb') as handle:
        pkl.dump(tfidf_vectorizer.vocabulary_, handle)
    
    lda_2 = LatentDirichletAllocation(n_components=n_topics, max_iter=n_iter, learning_method='batch',
                                      learning_offset=50., random_state=0, verbose=1)
    lda_2.fit(tfidf_train)
    
    tf_feature_names = tfidf_vectorizer.get_feature_names()
    if n_topics <= 10:
        plot_name = main_folder + dataset_name + '_lda_topic_tf_' + str(n_topics)
        plot_top_words(lda_2, tf_feature_names, n_top_words, plot_name)
    
    print('Extracting LDA with tf_idf features ...')
    
    tr_df = []
    for i in range(x_train.shape[0]):
        temp = []
        for j in range(x_train.shape[1]):
            temp.append(str(x_train[i, j]))
        xx = ' '.join(temp)
        tr_df.append(xx)
    
    X_train_vec = tfidf_vectorizer.transform(tr_df)
    X_train_topics = lda_2.transform(X_train_vec)
    
    te_df = []
    for i in range(x_test.shape[0]):
        temp = []
        for j in range(x_test.shape[1]):
            temp.append(str(x_test[i, j]))
        xx = ' '.join(temp)
        te_df.append(xx)
    X_test_vec = tfidf_vectorizer.transform(te_df)
    X_test_topics = lda_2.transform(X_test_vec)
    if not os.path.exists(main_folder + 'features/'):
        os.mkdir(main_folder + 'features/')
    
    np.save(main_folder + 'features/lda_tfidf_train', X_train_topics)
    np.save(main_folder + 'features/lda_tfidf_test', X_test_topics)


def get_top_n_words(n, n_topics, keys, document_term_matrix, count_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii',errors="ignore").decode('utf-8',errors="ignore"))
        top_words.append(" ".join(topic_words))
    return top_words


def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

