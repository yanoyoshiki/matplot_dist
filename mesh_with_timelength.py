from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from random import gauss
from re import S
from readline import insert_text
from cv2 import add
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import ipdb



class Data_dist():
        #二次元正規分布の確率密度を返す関数
    def gaussian(self,x,x_p,y_p):
        #2変数の分散共分散行列を指定
        # sigma=np.cov(x,y)
        sigma = np.array([[100,0],[0,100]])

        mu = np.array([x_p,y_p])
        # mu = np.array([1,1])
        #分散共分散行列の行列式
        det = np.linalg.det(sigma)
        # print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        # print(inv)
        
        return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    def baseline(self):
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape
        z = np.c_[X.ravel(),Y.ravel()]
        x=y=0
        #ここでベースを強制的に0にしている
        Z=self.gaussian(z,x,y)*0
        Z=Z.reshape(shape)
        return X,Y,Z
    
    def makeing_dataset(self,data1,data2,data3):
        data=[]
        data.append(data1)
        data.append(data2)
        data.append(data3)
        return data
    
    def insert(self,x_get_data_as_mean,y_get_data_as_mean):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        Z1_1 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)
        Z = Z1_1.reshape(shape)

        return X,Y,Z
    def multi_insert(self,x_get_data_as_mean,y_get_data_as_mean,multi_dim_data):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        # print("value_1 is {0} ,value_2 is {1} , value_3 is {2}".format(multi_dim_data[0],multi_dim_data[1],multi_dim_data[2]))
        # ipdb.set_trace()
        
        Z_1 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[0]
        Z_2 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[1]
        Z_3 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[2]
        Z_4 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[3]
        Z_5 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[4]
        Z_6 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[5]
        
        Z_1 = Z_1.reshape(shape)
        Z_2 = Z_2.reshape(shape)
        Z_3 = Z_3.reshape(shape)
        Z_4 = Z_4.reshape(shape)
        Z_5 = Z_5.reshape(shape)
        Z_6 = Z_6.reshape(shape)
        
        # ipdb.set_trace()
        return X,Y,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6
    
    def addition_distribute(self,Z_b1,Z_b2,Z_b3,Z_b4,Z_b5,Z_b6,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6):
        Z_1=Z_b1+Z_1
        Z_2=Z_b2+Z_2
        Z_3=Z_b3+Z_3
        Z_4=Z_b4+Z_4
        Z_5=Z_b5+Z_5
        Z_6=Z_b6+Z_6
        
        return Z_1,Z_2,Z_3,Z_4,Z_5,Z_6
    
    
    def making_time_width(self,Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6,r1,r2,r3,counts):       
        #r=[x_p,y_p,v1,v2,v3,v4,v5,v6]
        r=np.stack([r1,r2,r3])
        #Z_o_1[len(Z_o_1)-1]で最新の100*100が取れる
        if(counts==0):
            Z_o_1f,Z_o_2f,Z_o_3f,Z_o_4f,Z_o_5f,Z_o_6f=Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6
        else :
            Z_o_1f,Z_o_2f,Z_o_3f,Z_o_4f,Z_o_5f,Z_o_6f=Z_o_1[len(Z_o_1)-1],Z_o_2[len(Z_o_2)-1],Z_o_3[len(Z_o_3)-1],Z_o_4[len(Z_o_4)-1],Z_o_5[len(Z_o_5)-1],Z_o_6[len(Z_o_6)-1]
        
        # ipdb.set_trace()
        for i in range(3):
            X_m,Y_m,Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6=self.multi_insert(r[i,0],r[i,1],r[i,2:])
            # ipdb.set_trace()
            Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6=self.addition_distribute(Z_o_1f,Z_o_2f,Z_o_3f,Z_o_4f,Z_o_5f,Z_o_6f,Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6)
            Z_o_1f,Z_o_2f,Z_o_3f,Z_o_4f,Z_o_5f,Z_o_6f=Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6
        
        Z_o_1=np.block([[[Z_o_1]],[[Z_n_1]]])
        Z_o_2=np.block([[[Z_o_2]],[[Z_n_2]]])
        Z_o_3=np.block([[[Z_o_3]],[[Z_n_3]]])
        Z_o_4=np.block([[[Z_o_4]],[[Z_n_4]]])
        Z_o_5=np.block([[[Z_o_5]],[[Z_n_5]]])
        Z_o_6=np.block([[[Z_o_6]],[[Z_n_6]]])
        counts+=1
        
        return Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6,counts
    
if __name__ == "__main__":
    A=Data_dist()
    
    X_b,Y_b,Z_b=A.baseline()
    
    
    #sample data-----------
    x = np.random.randint(-100,100,100)
    y = np.random.randint(-100,100,100)
    tem = np.random.randint(-5,5,100)
    hum = np.random.randint(-5,5,100)
    nh3 = np.random.randint(-5,5,100)
    og = np.random.randint(-5,5,100)
    rg = np.random.randint(-5,5,100)
    bod = np.random.randint(-5,5,100)
    
        
        
    #generating robot sample data-----------
    x_p1=-50
    y_p1=50
    value_p1=[10,12,13,14,15,16]
    r1=[x_p1,y_p1]
    r1.extend(value_p1)
    # ipdb.set_trace()
    
    x_p2=10
    y_p2=20
    value_p2=[12,15,16,13,14,12]
    r2=[x_p2,y_p2]
    r2.extend(value_p2)
    
    x_p3=-10
    y_p3=-50
    value_p3=[16,15,12,13,14,12]
    r3=[x_p3,y_p3]
    r3.extend(value_p3)
    
    x_p4=-10
    y_p4=-50
    value_p4=[160,150,120,130,140,120]
    r4=[x_p4,y_p4]
    r4.extend(value_p4)
    
    data_n=A.makeing_dataset(r1,r2,r3)
    
    
    #--------------------------------
    X_m=Y_m=Z_b_1=Z_b_2=Z_b_3=Z_b_4=Z_b_5=Z_b_6=Z_b
    #サンプルデータで生成した分布をbase分布に載せる------
    r_b=np.stack([x.T,y.T,tem.T, hum.T,nh3.T,og.T,rg.T,bod.T], 1)
    # ipdb.set_trace()
    for i in range(len(x)):
        #最新と一個前を同時に扱う必要がある
        X_m,Y_m,Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6=A.multi_insert(r_b[i,0],r_b[i,1],r_b[i,2:])
        # ipdb.set_trace()
        Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6=A.addition_distribute(Z_b_1,Z_b_2,Z_b_3,Z_b_4,Z_b_5,Z_b_6,Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6)
        Z_b_1,Z_b_2,Z_b_3,Z_b_4,Z_b_5,Z_b_6=Z_n_1,Z_n_2,Z_n_3,Z_n_4,Z_n_5,Z_n_6
        # ipdb.set_trace()
    #-------------------------------------------------
    
    
    #start insert code
    #treating multidata is data_n
    X_m,Y_m,Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6=A.multi_insert(data_n[0][0],data_n[0][1],data_n[0][2:])
    X_m,Y_m,Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6=A.multi_insert(data_n[1][0],data_n[1][1],data_n[1][2:])
    X_m,Y_m,Z_3_1,Z_3_2,Z_3_3,Z_3_4,Z_3_5,Z_3_6=A.multi_insert(data_n[2][0],data_n[2][1],data_n[2][2:])
    
    Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6=A.addition_distribute(Z_b_1,Z_b_2,Z_b_3,Z_b_4,Z_b_5,Z_b_6,Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6)
    Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6=A.addition_distribute(Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6,Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6)
    Z_3_1,Z_3_2,Z_3_3,Z_2_4,Z_2_5,Z_2_6=A.addition_distribute(Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6,Z_3_1,Z_3_2,Z_3_3,Z_3_4,Z_3_5,Z_3_6)
    Zs1,Zs2,Zs3,Zs4,Zs5,Zs6=Z_3_1,Z_3_2,Z_3_3,Z_2_4,Z_2_5,Z_2_6
    #->前回分布と新規データを獲得したら起動する時系列ほｚ関数を作る必要がある
    
    
    #time_width-------------
    #generating robot sample data-----------
    x_p1=np.array([-50,-45,-40]).T
    y_p1=np.array([50,45,40]).T
    value_p1=np.array([[10,12,13,14,15,16],[10,12,13,14,15,16],[10,12,13,14,15,16]])
    p1=np.stack([x_p1,y_p1]).T
    rt1=np.block([p1, value_p1])
    # ipdb.set_trace()
    
    x_p2=np.array([10,15,20]).T
    y_p2=np.array([20,25,30]).T
    value_p2=np.array([[12,15,16,13,14,12],[12,15,16,13,14,12],[12,15,16,13,14,12]])
    p2=np.stack([x_p2,y_p2]).T
    rt2=np.block([p2, value_p2])
    
    x_p3=np.array([-10,-5,0]).T
    y_p3=np.array([-50,-55,60]).T
    value_p3=np.array([[16,15,12,13,14,12],[16,15,12,13,14,12],[16,15,12,13,14,12]])
    p3=np.stack([x_p3,y_p3]).T
    rt3=np.block([p3, value_p3])
    
    Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6=Zs1,Zs2,Zs3,Zs4,Zs5,Zs6
    # Z_o_1=Z_o_2=Z_o_3=Z_o_4=Z_o_5=Z_o_6=Z_b
    counts=0
    #ここでstackされた時系列分布行列を獲得することができる
    #試しにr達に時系列情報を持たせて複数時刻入力してみる(forの回す回数が時刻と思ったらいい)
    for i in range(len(rt1)):
    # for i in range(1):
        print(i)
        Z_tw_1,Z_tw_2,Z_tw_3,Z_tw_4,Z_tw_5,Z_tw_6,counts=A.making_time_width(Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6,rt1[i,:],rt2[i,:],rt3[i,:],counts)
        Z_o_1,Z_o_2,Z_o_3,Z_o_4,Z_o_5,Z_o_6=Z_tw_1,Z_tw_2,Z_tw_3,Z_tw_4,Z_tw_5,Z_tw_6
        # ipdb.set_trace()
    #-----------------------
    
    
    
    # ipdb.set_trace()
    #making graph
    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X_b, Y_b, Z_o_1[1])
    ipdb.set_trace()
    plt.show()