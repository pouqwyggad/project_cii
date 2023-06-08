from pymystem3 import Mystem
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics