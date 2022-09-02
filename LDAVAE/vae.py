from preprocessing import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
from plots import *
from imblearn.over_sampling import SMOTEN
from classifiers import *
disable_eager_execution()
sys.path.append('../')


class LVAE(object):

    def create2(self, max_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix):
        self.encoder = None
        self.decoder = None
        self.fnd = None
        self.autoencoder = None
        self.embedding_matrix = embed_matrix
        self.vocab_size = self.embedding_matrix.shape[0]
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.fnd_lambda = fnd_lambda
        input_txt = tf.keras.Input(shape=(max_length,), name='input_txt')
        encoded = self._build_encoder2(input_txt)

        self.encoder = tf.keras.Model(input_txt, outputs=encoded)
        encoded_input = tf.keras.Input(shape=(latent_dim,))
        predicted_outcome = self._build_fnd(encoded_input)
        self.fnd = tf.keras.Model(encoded_input, predicted_outcome)
        decoded_txt = self._build_decoder(encoded_input)

        self.decoder = tf.keras.Model(encoded_input, decoded_txt)

        self.autoencoder = tf.keras.Model(inputs=input_txt, outputs=[self.decoder(self.encoder(input_txt)), self._build_fnd(encoded)])

        losses = {"decoded_txt": "sparse_categorical_crossentropy", "fnd_output": "binary_crossentropy"}

        self.autoencoder.compile(optimizer=Adam(1e-5), loss=losses, metrics=['accuracy'],
                                 experimental_run_tf_function=False)

        self.get_features = K.function(input_txt, encoded)

    def _build_encoder2(self, input_txt):
        txt_embed = layers.Embedding(self.vocab_size, self.embedding_matrix.shape[1], input_length=self.max_length,
                                     name='txt_embed')(input_txt)
        lstm_txt_1 = layers.Bidirectional(layers.LSTM(self.latent_dim, return_sequences=True, name='lstm_txt_1',
                                                      activation='tanh',
                                                      kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)),
                                          merge_mode='concat')(txt_embed)

        fc_txt = layers.Dense(self.latent_dim, activation='relu', name='dense_txt',
                              kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(lstm_txt_1)
        h = layers.Dense(self.latent_dim, name='shared', activation='tanh',
                         kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(fc_txt)
        return h

    def _build_decoder2(self, encoded):
        dec_fc_txt = layers.Dense(self.latent_dim, name='dec_fc_txt', activation='tanh',
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(encoded)
        dec_lstm_txt_1 = layers.LSTM(self.latent_dim, return_sequences=True, activation='tanh', name='dec_lstm_txt_1',
                                     kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(dec_fc_txt)
        decoded_txt = layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'), name='decoded_txt')(
            dec_lstm_txt_1)
        return decoded_txt

    def _build_fnd2(self, encoded):
        h = layers.Dense(self.latent_dim*2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(self.fnd_lambda))(encoded)
        h = layers.Dense(self.latent_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.fnd_lambda))(h)
        # predicted_outcome = layers.Dense(1, activation='sigmoid', name='fnd_output')(h)
        return layers.Dense(1, activation='sigmoid', name='fnd_output')(h)

    def create(self, max_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix):
        self.encoder = None
        self.decoder = None
        self.fnd = None
        self.autoencoder = None
        self.embedding_matrix = embed_matrix
        self.vocab_size = self.embedding_matrix.shape[0]
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.fnd_lambda = fnd_lambda
        input_txt = tf.keras.Input(shape=(max_length,), name='input_txt')
        # vae_ce_loss, vae_mse_loss, encoded = self._build_encoder(input_txt)
        txt_loss, encoded = self._build_encoder(input_txt)

        self.encoder = tf.keras.Model(input_txt, outputs=encoded)
        # encoder.summary()
        encoded_input = tf.keras.Input(shape=(latent_dim,))
        predicted_outcome = self._build_fnd(encoded_input)
        self.fnd = tf.keras.Model(encoded_input, predicted_outcome)
        # fnd.summary()
        decoded_txt = self._build_decoder(encoded_input)

        self.decoder = tf.keras.Model(encoded_input, decoded_txt)

        decoder_output = self._build_decoder(encoded)

        self.autoencoder = tf.keras.Model(inputs=input_txt, outputs=[decoder_output, self._build_fnd(encoded)])

        # losses = {"decoded_txt": "sparse_categorical_crossentropy", "fnd_output": vae_mse_loss}
        losses = {"decoded_txt": "sparse_categorical_crossentropy", "fnd_output": "binary_crossentropy"}
        # losses = {"decoded_txt": txt_loss, "fnd_output": "binary_crossentropy"}

        # lossWeights = {"decoded_txt": 1.0, "fnd_output": 1.0}
        self.autoencoder.compile(optimizer=Adam(1e-5), loss=losses, metrics=['accuracy'],
                                 experimental_run_tf_function=False)
        # self.autoencoder.compile(optimizer=Adam(1e-5), loss=['sparse_categorical_crossentropy', vae_mse_loss],
        #                           metrics=['accuracy'])
        # self.autoencoder.compile(optimizer=Adam(1e-5), loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])

        self.get_features = K.function(input_txt, encoded)

    def _build_encoder(self, input_txt):
        txt_embed = layers.Embedding(self.vocab_size, self.embedding_matrix.shape[1], input_length=self.max_length,
                                     name='txt_embed', trainable=False, weights=[self.embedding_matrix])(input_txt)
        lstm_txt_1 = layers.Bidirectional(layers.LSTM(self.latent_dim, return_sequences=True, name='lstm_txt_1',
                                                      activation='tanh',
                                                      kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)),
                                          merge_mode='concat')(txt_embed)
        lstm_txt_2 = layers.Bidirectional(layers.LSTM(self.latent_dim, return_sequences=False, name='lstm_txt_2',
                                                      activation='tanh',
                                                      kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)),
                                          merge_mode='concat')(lstm_txt_1)
        fc_txt = layers.Dense(self.latent_dim, activation='tanh', name='dense_txt',
                              kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(
            lstm_txt_2)
        h = layers.Dense(self.latent_dim, name='shared', activation='tanh',
                         kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(fc_txt)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=0.01)
            return z_mean_ + K.exp(0.5 * z_log_var_) * epsilon

        z_mean = layers.Dense(self.latent_dim, name='z_mean', activation='linear')(h)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var', activation='linear')(h)

        # @tf.function
        def vae_mse_loss(x, x_decoded_mean):
            # mse_loss = objectives.mse(x, x_decoded_mean)
            # mse_loss = tf.keras.losses.MSE(x, x_decoded_mean)
            mse_loss = tf.keras.losses.mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return mse_loss + kl_loss

        # @tf.function
        def vae_ce_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
            xent_loss = K.binary_crossentropy(x, x_decoded_mean)

            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        def text_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            ce_loss = K.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return ce_loss + kl_loss

        return (text_loss,
                layers.Lambda(sampling, output_shape=(self.latent_dim,), name='lambda')([z_mean, z_log_var]))

    def _build_fnd(self, encoded):
        h = layers.Dense(self.latent_dim*2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(self.fnd_lambda))(encoded)
        h = layers.Dense(self.latent_dim, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(self.fnd_lambda))(h)
        # predicted_outcome = layers.Dense(1, activation='sigmoid', name='fnd_output')(h)
        return layers.Dense(1, activation='sigmoid', name='fnd_output')(h)

    def _build_decoder(self, encoded):
        dec_fc_txt = layers.Dense(self.latent_dim, name='dec_fc_txt', activation='tanh',
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(encoded)
        repeated_context = layers.RepeatVector(self.max_length)(dec_fc_txt)
        dec_lstm_txt_1 = layers.LSTM(self.latent_dim, return_sequences=True, activation='tanh', name='dec_lstm_txt_1',
                                     kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(repeated_context)
        dec_lstm_txt_2 = layers.LSTM(self.latent_dim, return_sequences=True, activation='tanh', name='dec_lstm_txt_2',
                                     kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda))(dec_lstm_txt_1)
        decoded_txt = layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'), name='decoded_txt')(
            dec_lstm_txt_2)
        # decoder = tf.keras.Model(encoded_input, decoded_txt, name="decoder")
        # decoder.summary()
        return decoded_txt


def train_vae(run_info, top_folder):
    print('Training VAE ...')
    main_folder = run_info['main_folder']
    sequence_length = run_info['sequence_length']
    latent_dim = run_info['latent_dim']
    reg_lambda = run_info['reg_lambda']
    fnd_lambda = run_info['fnd_lambda']
    epoch_no = run_info['epoch_no']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    text = np.load(this_data_folder + 'train_text_' + str(word2vec_dim) + '.npy')
    label = np.load(this_data_folder + 'train_label_' + str(word2vec_dim) + '.npy')[:, 1]
    embed_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')

    callback_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='fnd_output_loss', factor=0.2, patience=6,
                                                          min_lr=1e-7),
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.1, patience=1, verbose=0,
                                                      mode="auto", baseline=None, restore_best_weights=True)]
    model = LVAE()
    model.create(sequence_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix)

    model.autoencoder.fit(x=text, y={'decoded_txt': np.expand_dims(text, -1), 'fnd_output': label}, batch_size=64,
                          epochs=epoch_no, callbacks=callback_list, shuffle=True, validation_split=0.2,
                          use_multiprocessing=False)

    # save the model
    model.autoencoder.save_weights(main_folder + 'model_weights.hdf5')
    # model.autoencoder.save(path + '/weights/final_model_' + model_name)
    # tf.keras.models.save_model()
    model_history = model.autoencoder.history.history
    # model_config = model.autoencoder.get_config()

    with open(main_folder + 'model_history.pickle', 'wb') as handle:
        pkl.dump(model_history, handle)

    # with open(main_folder + 'model_config.pickle', 'wb') as handle:
    #     pkl.dump(model_config, handle)

    plot_history_with_2_outputs(main_folder, model_history)


def test_vae(run_info, top_folder, phase):
    main_folder = run_info['main_folder']
    sequence_length = run_info['sequence_length']
    latent_dim = run_info['latent_dim']
    reg_lambda = run_info['reg_lambda']
    fnd_lambda = run_info['fnd_lambda']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'
    test_text = np.load(this_data_folder + phase + '_text_' + str(word2vec_dim) + '.npy')
    test_label = np.load(this_data_folder + phase + '_label_' + str(word2vec_dim) + '.npy')[:, 1]
    embed_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')

    model = LVAE()
    model.create(sequence_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix)

    # load the model
    model.autoencoder.load_weights(main_folder + 'model_weights.hdf5')

    # Evaluate the model on test data
    pred = np.zeros([test_text.shape[0]])
    for tt in range(test_text.shape[0]):
        inp = test_text[tt:tt+1, :]
        pred[tt] = model.autoencoder.predict([inp])[-1]

    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    # print(accuracy_score(test_label, pred))
    # print(precision_recall_fscore_support(test_label, pred))
    return test_label, pred


def extract_features(run_info, top_folder, data_name):
    print('Extracting VAE features for', data_name)
    main_folder = run_info['main_folder']
    sequence_length = run_info['sequence_length']
    latent_dim = run_info['latent_dim']
    reg_lambda = run_info['reg_lambda']
    fnd_lambda = run_info['fnd_lambda']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    embed_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')

    # text_input = []
    # text_embed_input = []
    # text_embed_norm_input = []
    # output = []
    #
    # embedding_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')
    # embedding_matrix_norm = np.load(this_data_folder + 'embedding_matrix_norm_' + str(word2vec_dim) + '.npy')
    #
    # with open(this_data_folder + 'word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
    #     word_index = pkl.load(handle)
    #
    # for idx in range(len(filtered_data_df)):
    #     tweet_text = filtered_data_df.loc[idx, col_label]
    #     tweet_label = filtered_data_df.loc[idx, output_label]
    #     # if len(words) <= 2:
    #     #     continue
    #
    #     text = []
    #     text_embed = []
    #     text_embed_norm = []
    #     for word in words[:sequence_length]:
    #         if word in word_index:
    #             text.append(word_index[word])
    #         else:
    #             r = random.choice(list(word_index.values()))
    #             text.append(r)
    #         text_embed.append(embedding_matrix[text[-1]])
    #         text_embed_norm.append(embedding_matrix_norm[text[-1]])
    #
    #     while len(text) < sequence_length:
    #         text.append(0)
    #         text_embed.append(np.zeros(embedding_matrix.shape[1]))
    #         text_embed_norm.append(np.zeros(embedding_matrix_norm.shape[1]))
    #
    #     if tweet_label == 'fake':
    #         label = [0, 1]
    #     else:
    #         label = [1, 0]
    #
    #     text_input.append(text)
    #     text_embed_input.append(text_embed)
    #     text_embed_norm_input.append(text_embed_norm)
    #     output.append(label)
    #
    # test_text = np.array(text_input)
    test_text = np.load(this_data_folder + data_name + '_text_' + str(word2vec_dim) + '.npy')
    outputs = np.load(this_data_folder + data_name + '_label_' + str(word2vec_dim) + '.npy')[:, 1]
    # Save the Features
    # if not os.path.exists(path+'features'):
    #         os.makedirs(path+'features')
    model = LVAE()
    model.create(sequence_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix)

    model.autoencoder.load_weights(main_folder + 'model_weights.hdf5')

    # encoder_features = model.encoder.predict(test_text)
    # np.save(main_folder +'vae_encoder_features_' + data_name, encoder_features)

    learnt_features = np.zeros([test_text.shape[0], latent_dim])
    for i in range(test_text.shape[0]):
        text_batch = test_text[i:i+1]
        batch = model.get_features([text_batch])[0]
        learnt_features[i, :] = batch
    # np.save(main_folder + 'vae_learnt_features_' + data_name, learnt_features)
    # outputs = np.array(output)[:, 1]
    return learnt_features, outputs


def extract_features_over_sample(run_info, top_folder, filtered_data_df, data_name, col_label, output_label='label'):
    print('Extracting extended VAE features for', data_name)
    main_folder = run_info['main_folder']
    sequence_length = run_info['sequence_length']
    latent_dim = run_info['latent_dim']
    reg_lambda = run_info['reg_lambda']
    fnd_lambda = run_info['fnd_lambda']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    embed_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')

    text_input = []
    text_embed_input = []
    text_embed_norm_input = []
    output = []

    embedding_matrix = np.load(this_data_folder + 'embedding_matrix_' + str(word2vec_dim) + '.npy')
    embedding_matrix_norm = np.load(this_data_folder + 'embedding_matrix_norm_' + str(word2vec_dim) + '.npy')

    with open(this_data_folder + 'word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index = pkl.load(handle)

    for idx in range(len(filtered_data_df)):
        tweet_text = filtered_data_df.loc[idx, col_label]
        tweet_label = filtered_data_df.loc[idx, output_label]
        words = tokenize(tweet_text)
        # if len(words) <= 2:
        #     continue

        text = []
        text_embed = []
        text_embed_norm = []
        for word in words[:sequence_length]:
            if word in word_index:
                text.append(word_index[word])
            else:
                r = random.choice(list(word_index.values()))
                text.append(r)
            text_embed.append(embedding_matrix[text[-1]])
            text_embed_norm.append(embedding_matrix_norm[text[-1]])

        while len(text) < sequence_length:
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

    test_text = np.array(text_input)
    outputs = np.array(output)
    y_out = outputs[:, 1]
    sampler = SMOTEN(random_state=2021)
    text_res, label_res = sampler.fit_resample(test_text, y_out)
    np.save(main_folder + 'extented_text_' + data_name, text_res)
    # Save the Features
    # if not os.path.exists(path+'features'):
    #         os.makedirs(path+'features')
    model = LVAE()
    model.create(sequence_length, latent_dim, reg_lambda, fnd_lambda, embed_matrix)

    model.autoencoder.load_weights(main_folder + 'model_weights.hdf5')

    # encoder_features = model.encoder.predict(test_text)
    # np.save(main_folder +'vae_encoder_features_' + data_name, encoder_features)

    learnt_features = np.zeros([text_res.shape[0], latent_dim])
    for i in range(text_res.shape[0]):
        text_batch = text_res[i:i+1]
        batch = model.get_features([text_batch])[0]
        learnt_features[i, :] = batch
    # np.save(main_folder + 'vae_learnt_features_' + data_name, learnt_features)
    # outputs = np.array(output)[:, 1]

    return learnt_features, label_res


def lvae_classifier(run_info):
    main_folder = run_info['main_folder']
    # fnd_lambda = run_info['fnd_lambda']

    features_path = main_folder + 'features/'
    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tfidf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test.npy')

    y_train = np.load(main_folder + 'train_label.npy')
    y_test = np.load(main_folder + 'test_label.npy')

    if len(y_train.shape) == 2:
        y_train = y_train[:, 1]
    if len(y_test.shape) == 2:
        y_test = y_test[:, 1]

    vae_features_train, vae_features_test = scale_data(vae_features_train, vae_features_test)
    lda_features_train, lda_features_test = scale_data(lda_features_train, lda_features_test)
    features_tr = np.concatenate([vae_features_train, lda_features_train], axis=1)
    features_te = np.concatenate([vae_features_test, lda_features_test], axis=1)

    latent_space_dim = features_tr.shape[1]
    classifier = tf.keras.Sequential()
    # classifier.add(layers.Dense(latent_space_dim*2, activation='tanh',
    # kernel_regularizer=tf.keras.regularizers.l2(fnd_lambda), input_dim=latent_space_dim))
    classifier.add(layers.Dense(latent_space_dim, activation='relu'))
    classifier.add(layers.Dense(1, activation='sigmoid', name='fnd_output'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(features_tr, y_train, batch_size=128, epochs=70, verbose=1, validation_split=0.2)

    pred = classifier.predict(features_tr)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    pred_te = classifier.predict(features_te)
    pred_te[pred_te > 0.5] = 1
    pred_te[pred_te <= 0.5] = 0

    tr = compute_accuracy_metrics(y_train, pred.ravel().astype(int))
    te = compute_accuracy_metrics(y_test, pred_te.ravel().astype(int))
    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
    vae_results = pd.DataFrame(data=0, columns=['Value', 'Metric', 'Data'], index=range(12))
    vae_results.loc[0:5, 'Data'] = 'train'
    vae_results.loc[6:12, 'Data'] = 'test'
    vae_results.loc[0:5, 'Value'] = accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr
    vae_results.loc[6:12, 'Value'] = accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te
    vae_results.loc[[0, 6], 'Metric'] = 'Accuracy'
    vae_results.loc[[1, 7], 'Metric'] = 'Precision'
    vae_results.loc[[2, 8], 'Metric'] = 'Recall'
    vae_results.loc[[3, 9], 'Metric'] = 'F-Score'
    vae_results.loc[[4, 10], 'Metric'] = 'FPR'
    vae_results.loc[[5, 11], 'Metric'] = 'FNR'
    vae_results.to_csv(main_folder + 'MLP_relu_classifier_scaled_features.csv')


def vae_experiment(run_info, top_folder):
    
    main_folder = run_info['main_folder']
    
    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']
    
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'
    
    # x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    # x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')
    
    if not os.path.exists(main_folder + 'model_weights.hdf5') or \
            not os.path.exists(main_folder + 'model_history.pickle'):
        train_vae(run_info, top_folder)
    
    # lvae feature extraction
    if not os.path.exists(main_folder+'features/'):
        os.makedirs(main_folder+'features/')
    
    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/train_label.npy'):
        
        vae_features_tr, output_tr = extract_features(run_info, top_folder, data_name='train')
        
        np.save(main_folder + 'features/vae_train', vae_features_tr)
        np.save(main_folder + 'train_label', output_tr)
        
        # ext_x_train, ext_y_train = extract_features_over_sample(run_info, top_folder, x_train, data_name='train',
        #                                                   col_label=target_column, output_label='label')
        # np.save(main_folder + 'features/vae_ext_train', ext_x_train)
        # np.save(main_folder + 'train_ext_label', ext_y_train)
    
    if not os.path.exists(main_folder + 'features/vae_test.npy') or \
            not os.path.exists(main_folder + 'test_label.npy'):
        
        vae_features_te, output_te = extract_features(run_info, top_folder, data_name='test')
        
        np.save(main_folder + 'features/vae_test', vae_features_te)
        np.save(main_folder + 'test_label', output_te)
        
        # ext_x_test, ext_y_test = extract_features_over_sample(run_info, top_folder, x_test, data_name='test',
        #                                                   col_label=target_column, output_label='label')
        # np.save(main_folder + 'features/vae_ext_test', ext_x_test)
        # np.save(main_folder + 'test_ext_label', ext_y_test)
    
    # prediction
    y_tr, y_pred_tr = test_vae(run_info, top_folder, phase='train')
    y_te, y_pred_te = test_vae(run_info, top_folder, phase='test')
    
    tr_metrics = compute_accuracy_metrics(y_tr, y_pred_tr)
    te_metrics = compute_accuracy_metrics(y_te, y_pred_te)
    
    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr_metrics
    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te_metrics
    
    results = pd.DataFrame(data=0, columns=['Value', 'Metric', 'Data'], index=range(12))
    results.loc[0:5, 'Data'] = 'train'
    results.loc[6:12, 'Data'] = 'test'
    results.loc[0:5, 'Value'] = accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr
    results.loc[6:12, 'Value'] = accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te
    results.loc[[0, 6], 'Metric'] = 'Accuracy'
    results.loc[[1, 7], 'Metric'] = 'Precision'
    results.loc[[2, 8], 'Metric'] = 'Recall'
    results.loc[[3, 9], 'Metric'] = 'F-Score'
    results.loc[[4, 10], 'Metric'] = 'FPR'
    results.loc[[5, 11], 'Metric'] = 'FNR'
    results.to_csv(main_folder + 'model_classifier.csv')
