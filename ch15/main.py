# 畳み込み演算の実施
import numpy as np

def convld(x,w,p=0,s=1):
  w_rot = np.array(wp::-1)
  x_padded = np.array(x)
  if p>0:
    zero_pad = np.zeros(shape=p)