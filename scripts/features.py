from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy import sparse

import pandas as pd
import numpy as np

import re

class FeatureTransformer(BaseEstimator):
	def __init__(self):
		pass

	def get_feature_names(self):
		feature_names = []

		features_names.extend(self.truncated_svd.get_feature_names())
		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		summary = X.summary
		X['summary'] = X.summary.map(self.remove_periods)
		
 		summary_and_text = X.apply(self.combine_title_summary, axis=1)
		
		self.tfidf_vect = TfidfVectorizer(min_df=2, ngram_range=(1, 5), stop_words = None)
		
		summary_and_text_without_stopwords = map(self.remove_stopwords, summary_and_text)

		
		numerical_features = self.numerical_features(X, summary_and_text_without_stopwords, summary)
		
		tfidf_features = self.tfidf_vect.fit_transform(summary_and_text)
		self.truncated_svd = TruncatedSVD(n_components=140)
		truncated_svd_features = self.truncated_svd.fit_transform(tfidf_features)
		
		features = []
		features.append(numerical_features)
		features.append(truncated_svd_features)
		features = np.hstack(features)
		
		return features	

	def numerical_features(self, X, summary_and_text_without_stopwords, summary):
		num_authors = X.authors.map(lambda x: len(x.split(';')))
		num_tokens_in_text = [len(doc.split(' ')) for doc in summary_and_text_without_stopwords]
		num_sentences = [ len(x.split(';')) for x in summary]
		
		return np.array([num_authors, num_tokens_in_text, num_sentences]).T

	def combine_title_summary(self, X):
		summary = re.sub("[.]", " ", X['summary'])
		authors = re.sub("[;]", " ", X['authors'])
		title = X['title']

		return title + ' ' + authors + ' ' + summary

	def remove_stopwords(self, text):
		tokens = text.split(' ')
		rare_words = [token for token in tokens if not token.startswith('9') or not token.startswith('8')]
		
		return ' '.join(rare_words)

	def remove_periods(self, summary_text):
		return re.sub("[.]", " ", summary_text)

	def transform(self, X):
		summary = X.summary
		X['summary'] = X.summary.map(self.remove_periods)
		
 		summary_and_text = X.apply(self.combine_title_summary, axis=1)
		
		summary_and_text_without_stopwords = map(self.remove_stopwords, summary_and_text)
		
		numerical_features = self.numerical_features(X, summary_and_text_without_stopwords, summary)
		
		tfidf_features = self.tfidf_vect.transform(summary_and_text)
		truncated_svd_features = self.truncated_svd.transform(tfidf_features)
		
		features = []

		features.append(numerical_features)
		features.append(truncated_svd_features)
		features = np.hstack(features)
		
		return features