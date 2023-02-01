#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from attr import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class TfIdfParams(Config):
    """
    Configuration of TF-IDF vectorizer. For more details of parameters see 
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.

    :param input: ``{'filename', 'file', 'content'}``; If `'filename'`, the sequence passed as an argument to
        fit is expected to be a list of filenames that need reading to fetch the raw content to analyze.
    :param encoding: If bytes or files are given to analyze, this encoding is used to decode.
    :param decode_error: ``{'strict', 'ignore', 'replace'}``;
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    :param strip_accents: Remove accents and perform other character normalization
        during the preprocessing step.
    :param lowercase: Convert all characters to lowercase before tokenizing.
    :param preprocessor: Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    :param tokenizer: Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
    :param analyzer: Whether the feature should be made of word or character n-grams.
    :param stop_words: If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string value.
    :param token_pattern: Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``.
    :param ngram_range: The lower and upper boundary of the range of n-values for different
        n-grams to be extracted.
    :param max_df: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold.
    :param min_df: When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold.
    :param max_features: If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
    :param vocabulary: Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    :param binary: If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary.
    :param dtype: Type of the matrix returned by fit_transform() or transform().
    :param norm: Each output row will have unit norm, i.e., ``{'l1', 'l2'}``.
    :param use_idf: Enable inverse-document-frequency reweighting. If False, idf(t) = 1.
    :param smooth_idf: Smooth idf weights by adding one to document frequencies.
    :param sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    """

    input: str = "content"
    encoding: str = "utf-8"
    decode_error: str = "strict"
    strip_accents: object = None
    lowercase: bool = True
    preprocessor: object = None
    tokenizer: object = None
    analyzer: str = "word"
    stop_words: object = None
    token_pattern: str = r"(?u)\b\w\w+\b"
    ngram_range: tuple = (1, 1)
    max_df: float = 1.0
    min_df: int = 1
    max_features: object = None
    vocabulary: object = None
    binary: bool = False
    dtype: object = np.float64
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


@factory.register("vectorization", "tfidf", TfIdfParams)
class TfIdf(VectorizationAlgo):
    """
    TfIdf based vectorizer for log data. This is a wrapper class of the TF-IDF Vectorizer algorithm from scikit-learn.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.
    """

    def __init__(self, params: TfIdfParams, **kwargs):
        """
        Initializes TF-IDF vectorizer.

        :param params: TF-IDF algorithm parameters.
        :param kwargs: Optional k-v based params.
        """
        self.model = TfidfVectorizer(
            input=params.input,
            encoding=params.encoding,
            decode_error=params.decode_error,
            strip_accents=params.strip_accents,
            lowercase=params.lowercase,
            preprocessor=params.preprocessor,
            tokenizer=params.tokenizer,
            analyzer=params.analyzer,
            stop_words=params.stop_words,
            token_pattern=params.token_pattern,
            ngram_range=params.ngram_range,
            max_df=params.max_df,
            min_df=params.min_df,
            max_features=params.max_features,
            vocabulary=params.vocabulary,
            binary=params.binary,
            dtype=params.dtype,
            norm=params.norm,
            use_idf=params.use_idf,
            smooth_idf=params.smooth_idf,
            sublinear_tf=params.sublinear_tf,
        )

    def fit(self, loglines: pd.Series):
        """
        Trains a TF-IDF model.

        :param loglines: The input training dataset.
        """
        self.model.fit(loglines)
        self.vocab = self.model.vocabulary_
        self.vocab_size = len(self.vocab)

    def transform(self, loglines: pd.Series) -> pd.Series:
        """
        Transforms loglines into log vectors.

        :param loglines: The input test dataset.
        :return: The transformed log vectors.
        """
        res = self.model.transform(loglines)
        return pd.Series(res.todense().tolist(), index=loglines.index).apply(
            lambda x: np.array(x)
        )

    def summary(self):
        """
        Generates model summary.
        """
        return self.model.summary()
