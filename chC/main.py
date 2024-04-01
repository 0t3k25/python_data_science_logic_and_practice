# ベクトルの勉強少し
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['font.size'] =18
import numpy as np
x1 = np.array([[1], [0]])
x2 = np.array([[0], [1]])
# ベクトルを描画
plt.quiver([0,0],[0,0], np.hstack((x1,x2))[0,:],np.hstack((x1,x2))[1,:],
           color=['red', 'blue'], angles='xy', scale_units='xy', scale=1)

plt.text(0.5,-0.5, '$\\boldsymbol{x}_{1}$')
plt.text(-0.7,0.5,'$\\boldsymbol{x}_{2}$')
# 軸の範囲
plt.xlim(-3,3)
plt.ylim(-3,4.5)
# 軸の目盛りを設定
plt.xticks(np.arange(-3,3.1,1))
plt.yticks(np.arange(-3,4.1,1))
plt.grid()
ax=plt.gca()
# 横軸と縦軸のメモリの大きさを合わせる
ax.set(adjustable='box',aspect='equal')
plt.show()