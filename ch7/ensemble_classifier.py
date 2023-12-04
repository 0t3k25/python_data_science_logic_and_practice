# 多数決分類クラス作成
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_esrimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
  """多数決アンサンブル分類機

  パラメータ
  ------------
  classifiers : array-like, shape =[n_classifier]
    アンサンブルの様々な分類器
  
  vote : str {'classlabel', 'probability'} (default: 'classlabel')
    'classlabel'の場合、クラスラベルの予測はクラスラベルのargmaxに基づく
    'probability'の場合、クラスラベルの予測はクラスの所属確率の
    argmaxに基づく(分類器が調整済みである事が推奨される)
  
  weights : array-like,shape = [n_classifiers] (optional, default=None)
    `int`または`float`型の値のリストが提供された場合、分類器は重要度で重み付けされる
    `weights=None`の場合は均一な重みを使用
  """

  def __init__(self, classifiers,vote='classlabel',weights=None):

    self.classifiers=classifiers
    self.named_classifiers = {key: value for key,value in _name_estimators(classifiers)}
    self.vote = vote
    self.weights = weights

  def fit(self,X,y):
    """分類器を学習させる

       パラメータ
       -------------
       X: {array-like, sparse matrix}, shape = [n_examples, n_features]
        訓練データからなる行列
      
       y: array-like , shape = [n_examples]
       クラスラベルのベクトル
      
      戻り値
      -----------
      self : object
  """
  if self.vote not in ('probability', 'classlabel'):
    raise ValueError("vote must be 'probability'""or 'classlabel';got (vote%r)" %self.vote)

  if self.weights and len(self.weights) 