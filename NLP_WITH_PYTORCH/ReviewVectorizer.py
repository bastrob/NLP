import string
from collections import Counter

import numpy as np

from Vocabulary import Vocabulary


class ReviewVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, review_vocab, rating_vocab):
        """

        :param review_vocab: (Vocabulary): maps words to integers
        :param rating_vocab: (Vocabulary): maps class labels to integers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """
        Create a collapsed one-hot vector for the review
        :param review: (str): the review
        :return: one_hot (np.ndarray): the collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """
        Instantiate the vectorizer from the dataset dataframe
        :param review_df: (pandas.DataFrame): the review dataset
        :param cutoff: (int) the parameter for frequency-based filtering
        :return: an instance of the ReviewVectorizer
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # Add ratings
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # Add top words if count > provided count
        words_count = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    words_count[word] += 1

        for word, count in words_count.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """
        Instantiate a ReviewVectorizer from a serializable dictionary
        :param contents: (dict) the serializable dictionary
        :return: an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab, rating_vocab)

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        :return: the serializable dictionary
        """
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}