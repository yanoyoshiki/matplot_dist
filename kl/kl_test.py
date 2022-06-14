import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ipdb

# 正規分布
def gaussian1d(x,μ,delta):
    y = 1 / ( np.sqrt(2*np.pi* delta**2 ) )  * np.exp( - ( x - μ )**2  / ( 2 * delta ** 2 ) )
    return y

# 正規分布のKL divergence
def gaussian1d_KLdivergence(μ1,delta1,μ2,delta2):
    A = np.log(delta2/delta1)
    B = ( delta1**2 + (μ1 - μ2)**2 ) / (2*delta2**2)
    C = -1/2
    y = A + B + C
    return y

# KL divergence
def KLdivergence(p,q,dx):
    KL=np.sum(p * np.log(p/q)) * dx
    return KL


dx  = 0.01# xの刻み
xlm = [-6,6]# xの範囲
x   = np.arange(xlm[0],xlm[1]+dx,dx)# x座標
x_n   = len(x)# xの数

# Case 1
# p(x) = N(0,1)
# q(x) = N(μ,1)


μ1   = 0 # p(x)の平均μ1
delta1   = 1 # p(x)の標準偏差delta1  
px   = gaussian1d(x,μ1,delta1) # p(x)
delta2   = 1 # q(x)の標準偏差delta2
U2   = np.arange(-4,5,1)# q(x)の平均μ2
U2_n = len(U2)

# q(x)
Qx   = np.zeros([x_n,U2_n])

# KLダイバージェンス
KL_U2  = np.zeros(U2_n)

for i,μ2 in enumerate(U2):
    qx        = gaussian1d(x,μ2,delta2)
    Qx[:,i]   = qx
    ipdb.set_trace()
    KL_U2[i]  = KLdivergence(px,qx,dx)


# 解析解の範囲
U2_exc    = np.arange(-4,4.1,0.1)

# 解析解
KL_U2_exc = gaussian1d_KLdivergence(μ1,delta1,U2_exc,delta2)

# 解析解2
KL_U2_exc2 = U2_exc**2 / 2

# figure
fig = plt.figure(figsize=(8,4))
# デフォルトの色
clr=plt.rcParams['axes.prop_cycle'].by_key()['color']

# axis 1 
#-----------------------
# 正規分布のプロット
ax = plt.subplot(1,2,1)
# p(x)
plt.plot(x,px,label='$p(x)$')       
# q(x)
line,=plt.plot(x,Qx[:,i],color=clr[1],label='$q(x)$')       
# 凡例
plt.legend(loc=1,prop={'size': 13})

plt.xticks(np.arange(xlm[0],xlm[1]+1,2))
plt.xlabel('$x$')

# axis 2
#-----------------------
# KLダイバージェンス
ax2 = plt.subplot(1,2,2)
# 解析解
plt.plot(U2_exc,KL_U2_exc,label='Analytical')
# 計算
point, = ax2.plot([],'o',label='Numerical')

# 凡例
# plt.legend(loc=1,prop={'size': 15})

plt.xlim([U2[0],U2[-1]])
plt.xlabel('$\mu$')
plt.ylabel('$KL(p||q)$')

plt.tight_layout()

# 軸に共通の設定
for a in [ax,ax2]:
    plt.axes(a)
    plt.grid()
    # 正方形に
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())

plt.show()