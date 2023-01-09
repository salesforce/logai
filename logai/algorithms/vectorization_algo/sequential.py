import pandas as pd
from attr import dataclass
import pickle as pkl
import os

from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.utils.functions import pad
from logai.algorithms.factory import factory


@dataclass
class SequentialVectorizerParams(Config):
    """Config for Sequential Vectorizer which converts a sequence of loglines to sequence of log ids 

    Args:
       sep_token: separator token 
       model_save_dir: path to directory where models related to sequential vectorizer would be stored

    sep_token: str = None # separator token used to separate log lines in an input log sequence
    model_save_dir: str = None # path to directory where vectorizer models would be saved
    max_token_len: int = None # maximum token length of input

    """

    sep_token: str = None
    model_save_dir: str = None
    max_token_len: int = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)


@factory.register("vectorization", "sequential", SequentialVectorizerParams)
class Sequential(VectorizationAlgo):
    """ Sequential Vectorizer to convert a sequence of loglines to sequence of log ids 
    """

    def __init__(self, params: SequentialVectorizerParams):
        """Initializer for Sequential Vectorizer

        Args:
            params (SequentialVectorizerParams): config object for storing parameters of Sequential Vectorizer
        """
        self.params = params
        self.log_padding = "<pad>"
        self.log_oov = "<oov>"
        self.model_file = os.path.join(self.params.model_save_dir, "sequential.pkl")
        self.vocab = None
        if os.path.exists(self.model_file):
            self.vocab = pkl.load(open(self.model_file, "rb"))
            self.vocab_size = len(self.vocab)
            self.log_padding_id = self.vocab[self.log_padding]
            self.log_oov_id = self.vocab[self.log_oov]

    def _clean_data(self, data):
        return data.strip()

    def fit(self, loglines: pd.Series):
        """Fit method for training the sequential vectorizer

        Args:
            loglines (pd.Series):  pandas Series object containing the dataset on 
             which semantic vectorizer is trained (and the vocab is built). 
             Each data instance should be a logline or sequence of loglines concatenated by separator token 
        """
        model_file = None
        if self.params.model_save_dir:
            model_file = os.path.join(self.params.model_save_dir, "sequential.pkl")
        if os.path.exists(model_file):
            self.vocab = pkl.load(open(model_file, "rb"))
            self.vocab_size = len(self.vocab)
            self.log_padding_id = self.vocab[self.log_padding]
            self.log_oov_id = self.vocab[self.log_oov]
        else:
            loglines = loglines.apply(lambda x: self._clean_data(x))
            unique_loglines = set(loglines)
            unique_loglines.add(self.log_padding)
            unique_loglines.add(self.log_oov)
            self.vocab = {k: i for i, k in enumerate(list(unique_loglines))}
            self.vocab_size = len(self.vocab)
            self.log_padding_id = self.vocab[self.log_padding]
            self.log_oov_id = self.vocab[self.log_oov]
            if model_file:
                pkl.dump(self.vocab, open(model_file, "wb"))

    def transform(self, loglines: pd.Series) -> pd.Series:
        """Transform method for applying sequential vectorizer to loglines

        Args:
            loglines (pd.Series): pandas Series containing the data to be vectorized. 
            Each data instance should be a logline or sequence of loglines concatenated by separator token 

        Returns:
            pd.Series: vectorized loglines 
        """
        indices = loglines.index
        if self.params.sep_token is not None:
            loglines = loglines.apply(
                lambda x: pad(
                    [
                        self.vocab.get(self._clean_data(xi), self.log_oov_id)
                        for xi in x.split(self.params.sep_token)
                    ],
                    max_len=self.params.max_token_len,
                    padding_value=self.log_padding_id,
                )
            )
        loglines = pd.Series(loglines, indices)
        return loglines
