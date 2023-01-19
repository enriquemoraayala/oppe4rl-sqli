from collections import Counter
from .tokenizer import *
import numpy as np


class SqlFeatureExtractor(object):

    def __init__(self):
        # tokenizer_type = TokenizerType()  # this generates a 12 dimension feature vector
        tokenizer_tk = TokenizerTK()      # this generates a 702 dimension feature vector
        tokenizer_chr = TokenizerChr()    # this generates a 256 dimension feature vector
        self.tokenizers = [tokenizer_tk, tokenizer_chr]
        self.shape = (sum([tknz.vect_size for tknz in self.tokenizers]),)
        
    def extract(self, payload):
        feature_vector = np.array([])
        for tknz in self.tokenizers:
            new_feat = tknz.produce_feat_vector(payload, normalize=True)
            feature_vector = np.concatenate((feature_vector, new_feat))

        # normalize concatenated vector
        norm = np.linalg.norm(feature_vector)
        feature_vector = feature_vector / norm

        return feature_vector


class SqlSimpleFeatureExtractor(object):

    def __init__(self):
        # tokenizer_type = TokenizerType()  # this generates a 12 dimension feature vector
        max_token_len = 150
        self.tokenizer_tk = TokenizerTK(max_token_len=max_token_len)  # this generates a 702 dimension feature vector
        #tokenizer_chr = TokenizerChr()    # this generates a 256 dimension feature vector
     #   self.tokenizers = [tokenizer_tk]
        self.shape = (max_token_len,) #(sum([tknz.vect_size for tknz in self.tokenizers]),)

    def extract(self, payload):
        feature_vector = self.tokenizer_tk.produce_feat_vector(payload, normalize=True)

        return feature_vector


class SqlTermProportionFeatureExtractor(object):
    def __init__(self) -> None:
        self.tokenizer = TokenizerTK()
        self.shape = (self.tokenizer.vect_size,)

    def extract(self, payload):
        tokens = self.tokenizer.produce_feat_vector(payload, normalize=False)
        feature_vector = np.zeros(shape=self.shape)
        counter = Counter(tokens)
        for elem in counter:
            feature_vector[elem] = counter[elem]
        feature_vector = feature_vector / len(feature_vector)
        return feature_vector
        
class SqlEmbedFeatureExtractor(object):

    def __init__(self):

        self.embed_ = EmbedExtractor()      # this generates a 702 dimension feature vector

        self.shape = (768,) #(sum([tknz.vect_size for tknz in self.tokenizers]),)

    def extract(self, payload):
        feature_vector = self.embed_.produce_feat_vector(payload, normalize=True)

        return feature_vector
