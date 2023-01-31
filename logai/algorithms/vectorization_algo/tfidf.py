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
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    param: input: str = "content": {'filename', 'file', 'content'}
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.
    param: encoding: str = "utf-8": default='utf-8'
        - If bytes or files are given to analyze, this encoding is used to
        decode.
    param: decode_error: str = "strict": {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    param: strip_accents: object = None: {'ascii', 'unicode'} or callable
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        a direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    param: lowercase: bool = True
        Convert all characters to lowercase before tokenizing.
    param: preprocessor: object = None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.
    param: tokenizer: object = None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    param: analyzer: str = "word" {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

    param: stop_words: object = None: {'english'}, list
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
    param: token_pattern: str = r"(?u)\b\w\w+\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.
    param: ngram_range: tuple = (1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.
        Only applies if ``analyzer`` is not callable.
    param: max_df: float = 1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    param: min_df: int = 1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float in range of [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    param: max_features: object = None int
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.
    param: vocabulary: object = None Mapping or iterable
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    param: binary: bool = False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs).
    param: dtype: object = np.float64 default=float64
        Type of the matrix returned by fit_transform() or transform().
    param: norm: str = "l2" {'l1', 'l2'} or None
        Each output row will have unit norm, either:

        - 'l2': Sum of squares of vector elements is 1. The cosine
          similarity between two vectors is their dot product when l2 norm has
          been applied.
        - 'l1': Sum of absolute values of vector elements is 1.
          See :func:`preprocessing.normalize`.
        - None: No normalization.

    param: use_idf: bool = True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.
    param: smooth_idf: bool = True default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    param: sublinear_tf: bool = False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

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
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    def __init__(self, params: TfIdfParams, **kwargs):
        """
        Initialize TF-IDF vectorizer.
        :param params: TF-IDF algorithm parameters
        :param kwargs: Optional k-v based params
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
        Train TF-IDF model.
        :param loglines: pandas.Series input training set.
        :return:
        """
        self.model.fit(loglines)
        self.vocab = self.model.vocabulary_
        self.vocab_size = len(self.vocab)

    def transform(self, loglines: pd.Series) -> pd.Series:
        """
        Transform loglines into log vectors.
        :param loglines: pandas.Series input inference set.
        :return: pandas.Series
        """
        res = self.model.transform(loglines)
        return pd.Series(res.todense().tolist(), index=loglines.index).apply(
            lambda x: np.array(x)
        )

    def summary(self):
        """
        generate model summary
        :return: TfidfVectorizer.summary
        """
        return self.model.summary()
