# LDAVAE

## is an automated fake news detection method that incorporates two types of features from two models: 

### 1.   **VAE:** Deep neural embedding.
  *   **Motivation:** Efficiency
  *   Provides a lower dimensional semantic representation of the news article.
  *   Provides a lower dimensional semantic representation of the news article.


### 2.   **LDA:** Probabilistic topic modelling.
  *   **Motivation:** Interpretibility
  *   Provides topic-based features
  *   Provides a lower dimensional semantic representation of the news article.
  
---

## Contribution:

**(1)** We provide interpretability by incorporating Bayesian topic modelling and inferring topic compositions in news articles as added features for classification. 

**(2)** Our model works in the data scarcity scenario where only textual content is available.

**(3)** We keep our model efficient by coupling a deep architecture (VAE) to LDA.

---



## LDAVAE Notations:

 * $\mathcal{D}$: dataset, 
 * $\mathcal{D}_{tr}$: training set, 
 * $\mathcal{D}_{te}$: test set
 * $N$: Number of samples, indexed by $i$.
 * $V$: The set of vocabulary detected by word2vec.
 * $n_f$: Number of latent features obtained from encoder.
 * $w$: word2vec dimension.
 * $L = \max {l_i:i = 1,\dots, N }$
 * $l_i$: Length of sample $i$ (number of words).
 * $t_i^{(j)}$: Word $j$ in sample $i$.
 * $\lambda_1$: Regularization parameter ($=0.05$).
 * $\lambda_2$: Regularization parameter ($=0.3$).
 * $K$: Number of topics.

VAE: 

---

## VAE Structure:

### Encoder:
| Layer       | Output Shape        | Param \#          | Other Setting                                                        |
|-------------|---------------------|-------------------|----------------------------------------------------------------------|
| Input       | [(None, $L$)]       | 0                 |                               |
| Embedding   | (None, $L$, $w$)    | $\|V\| \times w $  | Non-trainable (word2vec)                                              |
| Bi. LSTM    | (None, $L$, $2n_f$) | $8n_f(w+n_f+1)  $ | activation='tanh'                        |
| Bi. LSTM    | (None, $2n_f$)      | $8n_f(3n_f+1)$    | activation='tanh'                         |
| Dense       | (None, $n_f$)       | $n_f(2n_f+1)$     | activation='tanh'                         |
| Dense ($h$) | (None, $n_f$)       | $n_f(n_f+1)$      | activation='tanh'                     |
| Sampling    | (None, $n_f$)       | 0                 |  $= \mu_h + \epsilon \exp(\sigma^2_h/2) $ |

### Decoder:
| Layer         | Output Shape        | Param \#       | Other Setting                                |
|---------------|---------------------|----------------|----------------------------------------------|
| Input         | [(None, $n_f$)] | 0              |                                              |
| Dense         | (None, $n_f$)       | $n_f(n_f+1)$   | activation='tanh' |
| Repeat Vector | (None, $L$, $n_f$)  | 0              |                   |
| LSTM          | (None, $L$, $n_f$)  | $4n_f(2n_f+1)$ | activation='tanh' |
| LSTM          | (None, $L$, $n_f$)  | $4n_f(2n_f+1)$ | activation='tanh' |
| Time Dist.    | (None, $L$, $\|V\|$)  | $\|V\|(n_f+1)$   | activation='softmax'|


### Classifier:
| Layer  | Output Shape    | Param \#      | Other Setting                                |
|--------|-----------------|---------------|----------------------------------------------|
| Input  | [(None, $n_f$)] | 0             |                                              |
| Dense  | (None, $2n_f$)  | $2n_f(n_f+1)$ | activation='tanh' |
| Dense  | (None, $n_f$)   | $n_f(2n_f+1)$ | activation='tanh' |
| Output | (None, 1)       | $n_f+1$       | activation='sigmoid'                         |

