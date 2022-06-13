from re import A, L
from black import mypyc_attr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
from inserdata_mesh import Data_dist

# 定義域[-1, 1]のx, yを50個区切りで生成


"""
data_dict={}
for i in range(10):
    for l in range(5):
        data_dict.setdefault("index_[{},{}]".format(i,l),[1,1])
"""

#座標を入れたらそれに対応する値を出力する
class Data_Dist():
    def __init__(self,dict):
        self.dict={}
        
    
    def data_pro_dist(d_1,d_2,d_3,d_4,d_5):
        np_stack=np.stack([d_1,d_2,d_3,d_4,d_5]).T
        return np_stack
    
    def dict_make(self,Distribution_data):
        count_num=0
        #辞書型の構築
        for i in range(10):
            for l in range(5):
                #座標ごとに多次元情報(npのlist)を振り分けている
                self.data_dict.setdefault("index_[{},{}]".format(i,l),Distribution_data[count_num,:])
            count_num+=1
            
    
    #位置に応じて値を出力するようにしている
    def response_d(self,x_p,y_p):
        print(self.data_dict["index_[{},{}]".format(x_p,y_p)])


#ロボットによる環境情報取得用コード(実験でもこれを用いる予定)
class Data_gat_by_robots():
    def __init__(self,dict):
        self.r_data_dict={}
        
    #全て平均値で構成される一様分布を作成
    #これを用いて情報量を算出する
    #space_sizeは空間の大きさ データ加工がめんどいので正方形にする
    def dict_make(self,space_size,Distribution_mean):
        for i in range(space_size):
            for l in range(space_size):
                self.r_data_dict.setdefault("index_[{},{}]".format(i,l),Distribution_mean)
    
    #xr_p,yr_pはロボットの自己位置推定結果
    #dataは取得した全センサーデータでありlist形式
    #データ取得してからそのデータが正規分布に従って生成されるようにする
    #その後平均分布に挿入して平均分布とのKL距離を測定して分布差を計算する
    def data_insert(self,xr_p,yr_p,data):
        self.r_data_dict["index_[{},{}]".format(xr_p,yr_p)]=data
        #正規分布生成
        #メッシュグリットの座標を使用して辞書に各センサーデータを格納する
        
        #取得データから分散小さめの正規分布を生成して平均値から引いたり足したりする量を正規分布によって決定する
        #取得データは離散的な値なので平均値と正規分布によって環境分布を平坦化する
        #辞書型変数にどんどん格納していく
    
    
    def one_dim_insert(self,x_get_data_as_mean,y_get_data_as_mean):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        Z = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)
        Z = Z.reshape(shape)

        return X,Y,Z
    
    
    #二次元正規分布の確率密度を返す関数
    def gaussian(self,x,x_p,y_p):
        #2変数の分散共分散行列を指定
        # sigma=np.cov(x,y)
        sigma = np.array([[100,0],[0,100]])

        mu = np.array([x_p,y_p])
        #分散共分散行列の行列式
        det = np.linalg.det(sigma)
        print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        print(inv)
        return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    def addition_distribute(self,Z_f,Z_s):
        Z_con=Z_f+Z_s

        return Z_con


if __name__ == "__main__":
    
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)

    tem = np.linspace(-1, 1, 50)
    hum = np.linspace(-1, 1, 50)
    og = np.linspace(-1, 1, 50)
    rg = np.linspace(-1, 1, 50)
    bod = np.linspace(-1, 1, 50)
    
    x_posi_1 = 10
    y_posi_1 = 10
    x_posi_2 = 20
    y_posi_2 = 20
    x_posi_3 = 30
    y_posi_3 = 30
    x_posi_4 = 40
    y_posi_4 = 40
    
    D=Data_Dist()
    Dr=Data_gat_by_robots()
    D_d=Data_dist()
    
    Dist_data=D.data_pro_dist(tem,hum,og,rg,bod)
    D.dict_make(Dist_data)
    
    D.response_d(x_posi_1,y_posi_1)
    
    #任意地点のセンサーデータセット
    gat_Dist_data_1=[10,10,10,10,10]
    gat_Dist_data_2=[20,20,20,20,20]
    gat_Dist_data_3=[30,30,30,30,30]
    gat_Dist_data_4=[40,40,40,40,40]
    
    
    
    
    
    
    X,Y,Z_1_1,Z_1_2,Z_1_3,Z_1_4=Dr.data_insert(x_posi_1,y_posi_1,gat_Dist_data_1)
    X,Y,Z_2_1,Z_2_2,Z_2_3,Z_2_4=Dr.data_insert(x_posi_2,y_posi_2)
    X,Y,Z_3_1,Z_3_2,Z_3_3,Z_3_4=Dr.data_insert(x_posi_3,y_posi_3)
    X,Y,Z_4_1,Z_4_2,Z_4_3,Z_4_4=Dr.data_insert(x_posi_4,y_posi_4)
    
    Z=Z_1+Z_2+Z_3+Z_4
    
    
    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X, Y, Z)
    # ipdb.set_trace()
    plt.show()
    ipdb.set_trace()


    """
    # index = np.arange(250).reshape(50,5)


    # Stack_ar=np.stack(tem,hum,og,rg,bod).T
    # for i in range(int(len(tem)/10)):
    #     A[:i*10,5]=index_num=np.stack([tem[:i*10],hum[:i*10],og[:i*10],rg[:i*10],bod[:i*10]]).T
    # A=index_num=np.stack([tem,hum,og,rg,bod],0).T
    # index[i,:]=index_num=np.stack([tem,hum,og,rg,bod],0).T
    
    
    # メッシュグリッドを生成
    xv, yv = np.meshgrid(x, y)

    # 関数x^2 - y^2の値をzに代入
    z = np.subtract(xv**2, yv**2)

    # x, y, zをワイヤフレームで表示
    # ax = Axes3D(plt.figure())
    # ax.plot_wireframe(xv, yv, z)
    plt.scatter(xv,yv)

    # plt.show()
    ipdb.set_trace()
    """