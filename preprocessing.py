import sys
import os
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
import gensim
from gensim.models import Word2Vec
import random
from sklearn.model_selection import train_test_split
import re
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from sklearn import feature_extraction
import unidecode
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTEN

warnings.filterwarnings('ignore')
sys.path.append('/')
abbr_list = ["n't", "'d", "'ll", "'s", "'m", "'ve", "'re"]
nltk.download('punkt')


def remove_punc(s):
    """ The function for removing the provided list of punctuations from a given string. """
    punctuation = '!"().:;>?{|},'
    return s.translate(str.maketrans('', '', punctuation))


def pre_process(address):
    file1 = open(address, 'r', encoding="utf8")
    Lines = file1.readlines()
    Lines = Lines[1:]
    data_tr = pd.DataFrame(data=0, columns=['text', 'label'], index=range(len(Lines)))

    count = 0
    # Strips the newline character
    for line_id in range(len(Lines)):
        line = Lines[line_id]
        sep_line = line.split('\t')
        if len(sep_line) == 7:
            text = sep_line[1]
            # text = remove_punc(text)
            # splitted = text.split(' ')
            # text = " ".join(list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, splitted)))
            data_tr.loc[line_id, :] = text, sep_line[6].split('\n')[0]
        if len(sep_line) == 6:
            text = sep_line[1]
            # text = remove_punc(text)
            # splitted = text.split(' ')
            # text = " ".join(list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, splitted)))
            data_tr.loc[line_id, :] = text, 'test'
        else:
            count += 1
    return data_tr, count


def remove_url(string):
    return re.sub(r'http\S+', '', string)


def process_text(string):
    txt = remove_url(string)
    txt = remove_parenthesis(txt)
    # txt = remove_stopwords(txt)
    txt = clean_fnc(txt)
    # txt = clean(txt)
    return txt


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(string):
    l = string.split(' ')
    return ' '.join([w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS])


def remove_parenthesis(sent):
    return ' '.join(sent.replace('(', ' ').replace(')', ' ').replace('.', '').split()).lower()


def clean_fnc(s):
    s = unidecode.unidecode(s) # for correct tokenization
    tokens = word_tokenize(s)
    for i, tok in enumerate(tokens):
        if tok not in abbr_list:
            tokens[i] = clean(tok)
    te = ' '.join(list(filter(lambda x: x != '', tokens))).lower()
    te.replace("_", " ")

    return te


def preprocess_and_filter(df, dataset_name, col_label='text'):
    if dataset_name == 'ISOT':
        new_df = pd.DataFrame(data=0, columns=[col_label], index=range(len(df)))

        filtered_idx = []
        # for idx, tweet in enumerate(tweets):
        for idx in range(len(df)):
            raw_tweet = df.loc[idx, col_label]
            tweet_text_list = process_text(raw_tweet)
            if len(tweet_text_list.split(' ')) <= 1 or tweet_text_list == ['0', '0']:
                filtered_idx.append(idx)
                new_df.loc[idx, col_label] = raw_tweet
            else:
                # tweet_text = ' '.join(tweet_text_list)
                if detect(tweet_text_list) != 'en':
                    filtered_idx.append(idx)
                new_df.loc[idx, col_label] = tweet_text_list

    elif dataset_name == 'Twitter':
        new_df = pd.DataFrame(data=0, columns=[col_label, 'label'], index=range(len(df)))
        filtered_idx = []
        # for idx, tweet in enumerate(tweets):
        for idx in range(len(df)):
            label = df.loc[idx, 'label']
            raw_tweet = df.loc[idx, col_label]
            tweet_text_list = process_text(raw_tweet)
            if len(tweet_text_list.split(' ')) <= 1 or tweet_text_list == ['0', '0']:
                filtered_idx.append(idx)
                new_df.loc[idx, :] = raw_tweet, label
            else:
                # tweet_text = ' '.join(tweet_text_list)
                if detect(tweet_text_list) != 'en':
                    filtered_idx.append(idx)
                new_df.loc[idx, :] = tweet_text_list, label

    elif dataset_name == 'Covid':
        new_df = pd.DataFrame(data=0, columns=[col_label, 'label'], index=range(len(df)))

        filtered_idx = []
        # for idx, tweet in enumerate(tweets):
        for idx in range(len(df)):
            raw_tweet = df.loc[idx, col_label]
            # label = 'fake' if df.loc[idx, 'outcome'] == 0 else 'real'
            label = df.loc[idx, 'outcome']
            tweet_text_list = process_text(raw_tweet)
            if len(tweet_text_list.split(' ')) <= 1:
                filtered_idx.append(idx)
                new_df.loc[idx, :] = tweet_text_list, label
            else:
                # tweet_text = ' '.join(tweet_text_list)
                if detect(tweet_text_list) != 'en':
                    filtered_idx.append(idx)
                new_df.loc[idx, :] = tweet_text_list, label

    # apply filtering
    filtered_data_df = new_df.drop(filtered_idx).reset_index(drop=True)
    return filtered_data_df, filtered_idx


def get_embeds(folder, data_df, embed_dim, col_label='text'):

    tweets = []
    for idx in range(len(data_df)):
        tweet_text = data_df.loc[idx, col_label]
        tweet_words = tweet_text.split()
        # if len(tweet_words) > 2:
        tweets.append(tweet_words)
    # tweets = list(data_df[col_label])
    print('Max sentence length:', max([len(tweet) for tweet in tweets]))
    print('Avg sentence length:', sum([len(tweet) for tweet in tweets]) / len(tweets))
    print('Min sentence length:', min([len(tweet) for tweet in tweets]))

    lent = [len(tweet) for tweet in tweets]
    # plot_doc_len_hist(lent)

    # for newer versions:
    # model = Word2Vec(tweets, min_count=1, vector_size=embed_dim)

    if gensim.__version__[0] == '4':
        model = Word2Vec(tweets, min_count=1, vector_size=embed_dim)

        words = list(model.wv.key_to_index.keys())
        word_index = {}
        ct = 0
        embedding_matrix = []
        # embedding_matrix.append(np.zeros(embed_dim))
        for word in words:
            if word not in word_index:
                word_index[word] = ct

                embedding_matrix.append(model.wv.get_vector(word))
                ct += 1
            else:
                print(word)

    else:
        # for gensim 3.8.3
        model = Word2Vec(tweets, min_count=1, size=embed_dim)
        words = list(model.wv.vocab)

        word_index = {}
        ct = 0
        embedding_matrix = []
        # embedding_matrix.append(np.zeros(embed_dim))
        for word in words:
            if word not in word_index:
                word_index[word] = ct
                embedding_matrix.append(model[word])
                ct += 1
            else:
                print(word)

    embedding_matrix = np.array(embedding_matrix)

    print("Vocab Size:", len(word_index))
    # run_info['n_features'] = len(word_index)
    # if not os.path.exists(folder + 'word2vec/'):
    #     os.makedirs(folder + 'word2vec/')
    with open(folder + 'word_index_' + str(embed_dim) + '.pkl', 'wb') as handle:
        pkl.dump(word_index, handle)
    # pkl.dump(word_index, open(folder + 'word2vec/word_index.pkl', 'w'))

    model.save(folder + 'word_embed_' + str(embed_dim) + '.bin')
    np.save(folder + 'embedding_matrix_' + str(embed_dim), np.array(embedding_matrix))

    max_val = np.max(embedding_matrix)
    min_val = np.min(embedding_matrix)
    embedding_matrix = (embedding_matrix - min_val) / (max_val - min_val)
    # for i in range(embed_dim):
    #     embedding_matrix[0][i] = 0
    np.save(folder + 'embedding_matrix_norm_' + str(embed_dim), np.array(embedding_matrix))


def load_data(folder, data_d, dataset_name, sequence_len, phase, embed_dim, col_label='text', output_label='label'):

    text_input = []
    text_embed_input = []
    text_embed_norm_input = []
    output = []
    # indices = []

    embedding_matrix = np.load(folder + 'embedding_matrix_' + str(embed_dim) + '.npy')
    embedding_matrix_norm = np.load(folder + 'embedding_matrix_norm_' + str(embed_dim) + '.npy')
    with open(folder + 'word_index_' + str(embed_dim) + '.pkl', 'rb') as handle:
        word_index = pkl.load(handle)

    for idx in range(len(data_d)):
        tweet_text = data_d.loc[idx, col_label]
        tweet_label = data_d.loc[idx, output_label]
        words = tweet_text.split()
        # if len(words) <= 2:
        #     continue
        text = []
        text_embed = []
        text_embed_norm = []
        for word in words[:sequence_len]:
            if word in word_index:
                text.append(word_index[word])
            else:
                r = random.choice(list(word_index.values()))
                text.append(r)
            text_embed.append(embedding_matrix[text[-1]])
            text_embed_norm.append(embedding_matrix_norm[text[-1]])

        while len(text) < sequence_len:
            text.append(0)
            text_embed.append(np.zeros(embedding_matrix.shape[1]))
            text_embed_norm.append(np.zeros(embedding_matrix_norm.shape[1]))

        if tweet_label == 'fake':
            label = [0, 1]
        else:
            label = [1, 0]

        text_input.append(text)
        text_embed_input.append(text_embed)
        text_embed_norm_input.append(text_embed_norm)
        output.append(label)

    text_input = np.array(text_input)
    text_embed_input = np.array(text_embed_input)
    text_embed_norm_input = np.array(text_embed_norm_input)
    output = np.array(output)
    print(output.shape[0])
    np.save(folder + phase + '_text_' + str(embed_dim), text_input)
    np.save(folder + phase + '_text_embed_' + str(embed_dim), text_embed_input)
    np.save(folder + phase + '_text_embed_norm_' + str(embed_dim), text_embed_norm_input)
    np.save(folder + phase + '_label_' + str(embed_dim), output)

    # use if the dataset is imbalance
    # if dataset_name == 'Covid':
    #     sampler = SMOTEN(random_state=2021)
    #     text_res, label_res = sampler.fit_resample(text_input, output[:, 1])
    #     output_2 = 1 - label_res
    #     labels = np.vstack((output_2, label_res)).T
    #     np.save(main_folder + phase + '_text_' + str(embed_dim), text_res)
    #     np.save(main_folder + phase + '_label_' + str(embed_dim), labels)


def text_2_np_covid(x_train, folder, word2vec_dim, sequence_len):

    with open(folder + 'word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index = pkl.load(handle)

    output = []
    text_input = []
    for idx in range(len(x_train)):
        tweet_text = x_train.loc[idx, 'headlines']
        tweet_label = x_train.loc[idx, 'label']
        words = tweet_text.split()
        text = []
        for word in words[:sequence_len]:
            if word in word_index:
                text.append(word_index[word])
            else:
                r = random.choice(list(word_index.values()))
                text.append(r)

        while len(text) < sequence_len:
            text.append(0)

        if tweet_label == 'fake':
            label = [0, 1]
        else:
            label = [1, 0]

        text_input.append(text)
        output.append(label)

    text_input = np.array(text_input)
    output = np.array(output)
    return text_input, output


def sample_np_covid(text_np_tr, output_tr_np, folder, word2vec_dim, sequence_len):
    from imblearn.over_sampling import SMOTEN
    sampler = SMOTEN(random_state=2021)
    text_res, label_res = sampler.fit_resample(text_np_tr, output_tr_np[:, 1])
    output_2 = 1 - label_res
    labels = np.vstack((output_2, label_res)).T

    with open(folder + 'word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index = pkl.load(handle)

    dictionary = {}
    for k, v in word_index.items():
        dictionary[v] = k

    new_x_train = pd.DataFrame(data=0, columns=['headlines', 'label'], index=range(text_res.shape[0]))

    for idx in range(text_res.shape[0]):
        words = [dictionary[text_res[idx, s]] for s in range(sequence_len)]
        tweet_text = ' '.join(words)
        new_x_train.loc[idx, 'headlines'] = tweet_text
        if labels[idx, 1] == 1:
            new_x_train.loc[idx, 'label'] = 'fake'
        else:
            new_x_train.loc[idx, 'label'] = 'real'


    return new_x_train, new_x_train['label']


def pre_process_dataset(folder, data_df_all, dataset_name, target_column, word2vec_dim, sequence_len, test_size=0.1):
    if dataset_name == 'ISOT':
        get_embeds(folder, data_df_all, embed_dim=word2vec_dim, col_label=target_column)
        real_no = len(data_df_all[data_df_all['label'] == 'real'])
        fake_no = len(data_df_all[data_df_all['label'] == 'fake'])
        print('Number of real posts:', real_no, '\nNumber of fake posts:', fake_no)

        x_train, x_test, y_train, y_test = train_test_split(data_df_all, data_df_all['label'],
                                                            test_size=test_size, random_state=2021,
                                                            shuffle=True, stratify=data_df_all['label'])
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        load_data(folder, x_train, dataset_name, sequence_len, phase='train', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')
        load_data(folder, x_test, dataset_name, sequence_len, phase='test', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')

    elif dataset_name == 'Covid':
        get_embeds(folder, data_df_all, embed_dim=word2vec_dim, col_label=target_column)
        real_no = len(data_df_all[data_df_all['label'] == 'real'])
        fake_no = len(data_df_all[data_df_all['label'] == 'fake'])
        print('Number of real posts:', real_no, '\nNumber of fake posts:', fake_no)

        x_train, x_test, y_train, y_test = train_test_split(data_df_all, data_df_all['label'], test_size=test_size,
                                                            random_state=2021, shuffle=True, stratify=data_df_all['label'])

        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        load_data(folder, x_train, dataset_name, sequence_len, phase='train', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')
        load_data(folder, x_test, dataset_name, sequence_len, phase='test', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')
        
        # temp_x_train = temp_x_train.reset_index(drop=True)
        # temp_x_test = temp_x_test.reset_index(drop=True)
        # temp_y_train = temp_y_train.reset_index(drop=True)
        # temp_y_test = temp_y_test.reset_index(drop=True)
        #
        # text_np_tr, output_tr_np = text_2_np_covid(temp_x_train, folder, word2vec_dim, sequence_len)
        # text_np_te, output_te_np = text_2_np_covid(temp_x_test, folder, word2vec_dim, sequence_len)
        #
        # x_train, y_train = sample_np_covid(text_np_tr, output_tr_np, folder, word2vec_dim, sequence_len)
        # x_test, y_test = sample_np_covid(text_np_te, output_te_np, folder, word2vec_dim, sequence_len)
        #
        # load_data(folder, temp_x_train, dataset_name, sequence_len, phase='train', embed_dim=word2vec_dim,
        # col_label=target_column, output_label='label')
        # load_data(folder, temp_x_test, dataset_name, sequence_len, phase='test', embed_dim=word2vec_dim,
        # col_label=target_column, output_label='label')

    elif dataset_name == 'Twitter':
        get_embeds(folder, data_df_all, embed_dim=word2vec_dim, col_label='text')
        x_train, x_test, y_train, y_test = train_test_split(data_df_all, data_df_all['label'],
                                                            test_size=test_size, random_state=2021,
                                                            shuffle=True, stratify=data_df_all['label'])

        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        load_data(folder, x_train, dataset_name, sequence_len, phase='train', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')
        load_data(folder, x_test, dataset_name, sequence_len, phase='test', embed_dim=word2vec_dim, col_label=target_column,
                  output_label='label')
    else:
        x_train, x_test, y_train, y_test = 0, 0, 0, 0

    return x_train, x_test, y_train, y_test


def prepare_data(run_info, top_folder, dataset_address):
    target_column = run_info['target_column']
    sequence_length = run_info['sequence_length']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    print('Preparing data ...')
    data_folder = top_folder + 'data/'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    this_data_folder = data_folder + dataset_name + '/'
    if not os.path.exists(this_data_folder):
        os.makedirs(this_data_folder)

    if dataset_name == 'ISOT':
        real_address = dataset_address + dataset_name + '/True.csv'
        fake_address = dataset_address + dataset_name + '/Fake.csv'
        data_df_real = pd.read_csv(real_address)
        data_df_fake = pd.read_csv(fake_address)
        filtered_data_df_real, _ = preprocess_and_filter(data_df_real, dataset_name, col_label=target_column)
        filtered_data_df_fake, _ = preprocess_and_filter(data_df_fake, dataset_name, col_label=target_column)
        filtered_data_df_real['label'] = 'real'
        filtered_data_df_fake['label'] = 'fake'
        filtered_data_df_all = filtered_data_df_real.append(filtered_data_df_fake, ignore_index=True)

        x_train, x_test, y_train, y_test = pre_process_dataset(this_data_folder, filtered_data_df_all, dataset_name,
                                                               target_column, word2vec_dim, sequence_len=sequence_length,
                                                               test_size=0.1)
    elif dataset_name == 'Covid':
        # this data is the original imbalance data
        # address = dataset_address + dataset_name + '_imbalanced' + '/COVID_Fake_News_Data.csv'
        address = dataset_address + dataset_name + '/covidSelfDataset.csv'
        data_df = pd.read_csv(address)
        filtered_data_df, _ = preprocess_and_filter(data_df, dataset_name, col_label=target_column)

        x_train, x_test, y_train, y_test = pre_process_dataset(this_data_folder, filtered_data_df, dataset_name,
                                                               target_column, word2vec_dim, sequence_len=sequence_length,
                                                               test_size=0.1)

    elif dataset_name == 'Twitter':
        tr_data_address = dataset_address + dataset_name + '/posts_tr.txt'
        # te_data_address = dataset_address + dataset_name + '/posts_te.txt'
        tr_data, c = pre_process(tr_data_address) # no text preprocessing is done here, only makes the df
        # te_data, _ = pre_process(te_data_address)
        filtered_data_df_tr, gg = preprocess_and_filter(tr_data, dataset_name, col_label=target_column)
        # filtered_data_df_te, cc = preprocess_and_filter(te_data, dataset_name, col_label=target_column)
        # filtered_data_df_all = filtered_data_df_tr.append(filtered_data_df_te, ignore_index=True)
        filtered_data_df_all = filtered_data_df_tr
        x_train, x_test, y_train, y_test = pre_process_dataset(this_data_folder, filtered_data_df_all, dataset_name,
                                                               target_column, word2vec_dim, sequence_len=sequence_length,
                                                               test_size=0.1)
    else:
        print('invalid dataset.')
        x_train, x_test, y_train, y_test = 0, 0, 0, 0

    # save the data

    x_train.to_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv', index=False)
    x_test.to_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv', index=False)
    y_train.to_csv(this_data_folder + 'y_train_' + str(word2vec_dim) + '.csv', index=False)
    y_test.to_csv(this_data_folder + 'y_test_' + str(word2vec_dim) + '.csv', index=False)


def make_run_info(top_folder, dataset_name, latent_dim, epoch_no, n_topics, n_iter, word2vec_dim):
    # vae parameters
    reg_lambda = 0.05
    fnd_lambda = 0.3

    # lda parameters
    n_features = 20000
    n_top_words = 20
    included_features = ['vae', 'lda_tfidf'] # ['vae', 'lda_tf']

    if dataset_name == 'ISOT':
        sequence_length = 36 # 36 with title, 8278 with the text
        target_column = 'title'  # 'title', 'text'

    elif dataset_name == 'Covid':
        sequence_length = 115 # 61
        target_column = 'text' # 'headlines'  # 'title', 'text'

    elif dataset_name == 'Twitter':
        sequence_length = 31
        target_column = 'text'  # 'title', 'text'

    else:
        sequence_length = 0
        target_column = ''  # 'title', 'text'

    model_name = dataset_name + '_ep_' + str(epoch_no) + '_seq_len_' + str(sequence_length) + '_latent_dim_' + \
                 str(latent_dim) + '_w2v_' + str(word2vec_dim)

    if not os.path.exists(top_folder):
        os.makedirs(top_folder)

    main_folder = top_folder + model_name + '/'

    run_info = {
        'main_folder': main_folder,
        'sequence_length': sequence_length,
        'latent_dim': latent_dim,
        'reg_lambda': reg_lambda,
        'fnd_lambda': fnd_lambda,
        'epoch_no': epoch_no,
        'model_name': model_name,
        'n_topics': n_topics,
        'target_column': target_column,
        'n_top_words': n_top_words,
        'n_features': n_features,
        'n_iter': n_iter,
        'included_features': included_features,
        'word2vec_dim': word2vec_dim,
        'dataset_name': dataset_name,
    }

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    with open(main_folder + 'run_info.pickle', 'wb') as handle:
        pkl.dump(run_info, handle)

    file1 = open(main_folder + "run_info.txt", "w")  # write mode
    for k in run_info.keys():
        print(k, ':', run_info[k])
        file1.write(k + ':' + str(run_info[k]) + "\n")
    file1.close()

    return run_info


