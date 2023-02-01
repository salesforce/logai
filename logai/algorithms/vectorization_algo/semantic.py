#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import gensim
import numpy as np
import pandas as pd
import string
import pickle as pkl
import os
import gensim.downloader
import logging
from attr import dataclass
from nltk.tokenize import word_tokenize

from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.utils.functions import pad
from logai.algorithms.factory import factory


@dataclass
class SemanticVectorizerParams(Config):
    """
    Configuration of Semantic vectorization of loglines (or sequence of log lines) using models like word2vc, glove and fastText.

    :param max_token_len: maximum token length of the input.
    :param min_token_count: minimum count of occurrences of a token in training data for it to be considered in the vocab.
    :param sep_token: separator token used to separate log lines in input log sequence. Default is "[SEP]".
    :param embedding_dim: embedding dimension of the learnt token embeddings.
    :param window: window size parameter for word2vec and fastText models.
    :param embedding_type: type of embedding, currently supports glove, word2vec and fastText. Default is "fasttext".
    :param model_save_dir: path to directory where vectorizer models would be saved.
    """

    max_token_len: int = 10
    min_token_count: int = 1
    sep_token: str = "[SEP]"
    embedding_dim: int = 300
    window: int = 3
    embedding_type: str = "fasttext"
    model_save_dir: str = None


@factory.register("vectorization", "semantic", SemanticVectorizerParams)
class Semantic(VectorizationAlgo):
    """
    Semantic vectorizer to convert loglines into token ids based on a embedding model and vocabulary
    (like word2vec, glove and fastText). It supports either pretrained models and pretrained vocabulary
    or training word embedding models like Word2Vec or FastText on the given training data.

    :param params: A config object for semantic vectorizer.
    """

    def __init__(self, params: SemanticVectorizerParams):
        
        self.params = params
        self.model = None
        self.vocab = None
        self.vocab_size = None
        self.embed_matrix = None
        self.vocab_filename = None
        self.embed_mat_filename = None
        if self.params.model_save_dir:
            self.embed_mat_filename = os.path.join(
                self.params.model_save_dir, "embedding_matrix.npy"
            )
            self.vocab_filename = os.path.join(self.params.model_save_dir, "vocab.pkl")

        self.train_embedding_model = False

        if os.path.exists(self.vocab_filename) and os.path.exists(
            self.embed_mat_filename
        ):
            self.vocab = pkl.load(open(self.vocab_filename, "rb"))
            self.embed_matrix = np.load(self.embed_mat_filename)
            self.params.embedding_dim = self.embed_matrix.shape[1]
            self.vocab_size = len(self.vocab)

    def _tokenize_logline(self, sentence):
        try:
            sentence = sentence.translate(
                str.maketrans(string.punctuation, " " * len(string.punctuation))
            )
        except Exception as e:
            logging.info("Cannot process line: {} ".format(sentence))
            sentence = ""
        token_list = word_tokenize(sentence.lower())
        return token_list

    def fit(self, loglines: pd.Series):
        """Fit method to train semantic vectorizer.

        :param loglines: A pandas Series object containing the dataset on
            which semantic vectorizer is trained (and the vocab is built).
            Each data instance should be a logline or sequence of loglines
            concatenated by separator token.
        """
        if (
            self.params.model_save_dir
            and os.path.exists(self.vocab_filename)
            and os.path.exists(self.embed_mat_filename)
        ):
            self.vocab = pkl.load(open(self.vocab_filename, "rb"))
            self.embed_matrix = np.load(self.embed_mat_filename)
            self.params.embedding_dim = self.embed_matrix.shape[1]
            self.vocab_size = len(self.vocab)
        else:
            doc = []
            for sentence in loglines:
                token_list = self._tokenize_logline(sentence)
                doc.extend(token_list)

            doc_words = set(doc)
            if self.params.embedding_type.lower() == "glove":
                # Use Word2Vec for vectorization
                if self.train_embedding_model:
                    self.model = gensim.models.Word2Vec(
                        doc,
                        min_count=self.params.min_token_count,
                        vector_size=self.params.embedding_dim,
                        window=self.params.window,
                    )
                else:
                    if self.params.embedding_dim in [50, 100, 200, 300]:
                        self.model = gensim.downloader.load(
                            "glove-wiki-gigaword-" + str(self.embed_dim)
                        )
                    else:
                        raise ValueError(
                            "embedding dim supported for glove pretrained model\
                                 is any of (50, 100, 200, 300)"
                        )

            elif self.params.embedding_type.lower() == "word2vec":
                if self.train_embedding_model:
                    self.model = gensim.models.Word2Vec(
                        doc,
                        min_count=self.params.min_token_count,
                        vector_size=self.params.embedding_dim,
                        window=self.params.window,
                    )
                else:
                    if self.params.embedding_dim != 300:
                        raise ValueError(
                            "embedding dim supported for word2vec pretrained model is 300"
                        )
                    self.model = gensim.downloader.load("word2vec-google-news-300")

            elif self.params.embedding_type.lower() == "fasttext":
                if self.train_embedding_model:
                    self.model = gensim.models.FastText(
                        doc,
                        min_count=self.params.min_token_count,
                        vector_size=self.params.embedding_dim,
                        window=self.params.window,
                    )
                else:
                    if self.params.embedding_dim != 300:
                        raise ValueError(
                            "embedding dim supported for fasttext pretrained model is 300"
                        )
                    self.model = gensim.downloader.load(
                        "fasttext-wiki-news-subwords-300"
                    )

            if self.train_embedding_model:
                word_vectors = self.model.wv
                zero_vectors = np.zeros((3, self.params.embedding_dim))
                word_vectors.add_vectors(
                    ["UNK", "PAD", self.params.sep_token], zero_vectors
                )
                self.vocab = {k: i for i, k in enumerate(word_vectors.index_to_key)}
                self.embed_matrix = word_vectors.vectors
                self.vocab_size = len(self.vocab)
                if self.vocab_filename:
                    pkl.dump(self.vocab, open(self.vocab_filename, "wb"))
                if self.embed_mat_filename:
                    np.save(self.embed_mat_filename, self.embed_matrix)
            else:
                zero_vectors = np.zeros((3, self.params.embedding_dim))
                doc_words.update(["UNK", "PAD", self.params.sep_token])
                self.model.add_vectors(
                    ["UNK", "PAD", self.params.sep_token], zero_vectors
                )
                doc_words = doc_words.intersection(set(self.model.index_to_key))
                word_vectors_map = {k: i for i, k in enumerate(self.model.index_to_key)}
                doc_words_indices = [word_vectors_map[w] for w in doc_words]
                self.embed_matrix = self.model.vectors[doc_words_indices]
                self.vocab = {k: i for i, k in enumerate(doc_words)}
                self.vocab_size = len(self.vocab)
                if self.vocab_filename:
                    pkl.dump(self.vocab, open(self.vocab_filename, "wb"))
                if self.embed_mat_filename:
                    np.save(self.embed_mat_filename, self.embed_matrix)

    def transform(self, loglines: pd.Series) -> pd.Series:
        """Transform method to run semantic vectorizer on loglines.

        :param loglines: The pandas Series containing the data to be vectorized.
            Each data instance should be a logline or sequence of loglines concatenated by separator token.
        :return: The vectorized log data.
        """
        log_vectors = []
        count = 0
        for ll in loglines:
            token_list = self._tokenize_logline(ll)
            token_ids = [self.vocab.get(t, self.vocab["UNK"]) for t in token_list]
            log_vector = pad(
                np.array(token_ids),
                self.params.max_token_len,
                padding_value=self.vocab["PAD"],
            )

            log_vectors.append(log_vector)
            count += 1
        log_vector_series = pd.Series(log_vectors, index=loglines.index)
        logging.info("Finished converting loglines to token ids")
        return log_vector_series

    def summary(self):
        """
        Generate model summary.
        """
        return self.model.summary()
