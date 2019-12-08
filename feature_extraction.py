#==========================================
# Title: Feature extraction
# Author: Rajesh Gupta
# Date:   16 Nov 2019
#==========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

class FeatureExtractor:

	def __init__(self, df=None, input_label=None, output_label=None):
		self._df = df
		self._input_label = input_label
		self._output_label = output_label
		self.input_model_trained = False
		self.output_model_trained = False

	def initialize_tf_idf(self):
		self._tf_idf = TfidfVectorizer()
		
	def initialize_multi_label_linarizer(self):
		self._mlb = MultiLabelBinarizer()

	@property
	def tf_idf(self):
		return self._tf_idf
	
	@property
	def mlb(self):
		return self._mlb
	
	@property
	def df(self):
		return self._df

	@property
	def input_label(self):
		return self._input_label
	
	@property
	def output_label(self):
		return self._output_label

	def fit_input(self, kind="tf-idf"):
		if kind == "tf-idf":
			if not self.input_model_trained:
				self.initialize_tf_idf()
				self.input_model_trained = True
			self._tf_idf.fit(self._df[self._input_label])

	def transform_input(self, df=None, column=None):
		if not df:
			df = self._df
			column = self._input_label
		return self._tf_idf.transform(df[column])

	def fit_output(self, kind="mlb"):
		if kind == "mlb":
			if not self.output_model_trained:
				self.initialize_multi_label_linarizer()
				self.output_model_trained = True
			self._mlb.fit(self._df[self._output_label])

	def transform_output(self, df=None, column=None):
		if not df:
			df = self._df
			column = self._output_label
		return self._mlb.transform(df[column])
