# これは時系列情報も保有している分布を取り込む
# 時系列的にも比較して3つ分の情報量が高い箇所を取り出した物
from dis import dis
from cv2 import transform
import torch
import torch.nn.functional as F
import numpy as np
import ipdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mesh_with_timelength import Data_dist


class KL_dist():
    # def gaussian(self,x,x_p,y_p):
    #     #2変数の分散共分散行列を指定
    #     # sigma=np.cov(x,y)
    #     sigma = np.array([[100,0],[0,100]])

    #     mu = np.array([x_p,y_p])
    #     # mu = np.array([1,1])
    #     #分散共分散行列の行列式
    #     det = np.linalg.det(sigma)
    #     # print(det)
    #     #分散共分散行列の逆行列
    #     inv = np.linalg.inv(sigma)
    #     n = x.ndim
        
    #     return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    # def baseline(self):
    #     x = np.linspace(-100, 100, 100)
    #     y = np.linspace(-100, 100, 100)
    #     X, Y = np.meshgrid(x, y)
    #     shape = X.shape
    #     z = np.c_[X.ravel(),Y.ravel()]
    #     x=y=0
    #     #ここでベースを強制的に0にしている
    #     Z=self.gaussian(z,x,y)*0
    #     Z=Z.reshape(shape)
    #     return X,Y,Z
        
    def torch_KL(self,first_dist,second_dist):
        torch_KL_d_value=F.kl_div(first_dist.log(), second_dist, None, None, 'sum')
        return torch_KL_d_value

    def sprit_calcu_sensor_time(self,insert_dist,av_dist,sensor_index,time_index):
        #ここで獲得可能な分布の形状は(センサー種，時系列情報，x座標，y座標)になっている。
        #つまり(6,対象時刻分の長さ,100,100)となる
        
        KL_list=[]
        print("KL list setting")
        #それぞれのセンサー値取得後の区画分布と平均分布の区画分布のKLを算出
        #区画毎にKLを出力
        
        #======================
        # #------------------------
        #センサー種，時刻指定
        #------------------------
        print(int(len(av_dist[0,0,:,:])/10))
        for i in range(int(len(av_dist[0,0,:,:])/10)):
            for l in range(int(len(av_dist[0,0,:,:])/10)):
                KL_value=self.torch_KL(insert_dist[sensor_index,time_index,i*10:i*10+10,l*10:l*10+10]+1
                                       ,av_dist[sensor_index,time_index,i*10:i*10+10,l*10:l*10+10]+1)
                print("{} to {} dist shape is {}".format(i*10,i*10+10,insert_dist[0,0,i*10:i*10+10,l*10:l*10+10].shape))
                print("{} and {}".format(i,l))
                KL_list.append(KL_value)
                ipdb.set_trace()
        #======================
        
        disappend_index_list=[]
        #その中でKLが高い順で3つとってくる
        KL_array=np.array(KL_list)
        # ipdb.set_trace()
        for i in range(3):
            disappend_index=np.argmax(KL_array)
            KL_array=np.delete(KL_array,disappend_index)
            disappend_index_list.append(disappend_index)
            # ipdb.set_trace()
        # ipdb.set_trace()
        print("No.1 is {}, No.2 is {}, No.3 is {}".format(disappend_index_list[0],disappend_index_list[1]+1,disappend_index_list[2]+2))
        
        
        print("now codeing")
    
    
    def sprit_calcu_time(self,insert_dist,av_dist,time_index):
        KL_list=[]
        #======================
        #------------------------
        #時刻のみ指定
        #------------------------
        print(int(len(av_dist[0,0,:,:])/10))
        for i in range(int(len(av_dist[0,0,:,:])/10)):
            for l in range(int(len(av_dist[0,0,:,:])/10)):
                KL_value=self.torch_KL(insert_dist[:,time_index,i*10:i*10+10,l*10:l*10+10]+1
                                       ,av_dist[:,time_index,i*10:i*10+10,l*10:l*10+10]+1)
                print("{} to {} dist shape is {}".format(i*10,i*10+10,insert_dist[0,0,i*10:i*10+10,l*10:l*10+10].shape))
                print("{} and {}".format(i,l))
                KL_list.append(KL_value)
                # ipdb.set_trace()
        #======================
        
        disappend_index_list=[]
        #その中でKLが高い順で3つとってくる
        KL_array=np.array(KL_list)
        space_KL=KL_array.reshape(10,10)
        
        
        # ipdb.set_trace()
        for i in range(3):
            disappend_index=np.argmax(KL_array)
            KL_array=np.delete(KL_array,disappend_index)
            disappend_index_list.append(disappend_index)
            # ipdb.set_trace()
        # ipdb.set_trace()
        print("No.1 is {}, No.2 is {}, No.3 is {}".format(disappend_index_list[0],disappend_index_list[1]+1,disappend_index_list[2]+2))
        
        return disappend_index_list[0],disappend_index_list[1]+1,disappend_index_list[2]+2,space_KL
        
    def position_KL_calcu(self,space_KL,p_x,p_y):
        # ipdb.set_trace()
        position_KL=space_KL[p_x,p_y]
        return position_KL

    def sprit_calcu_all(self,insert_dist,av_dist):
        KL_list=[]
        #======================
        #------------------------
        #時刻のみ指定
        #------------------------
        print(int(len(av_dist[0,0,:,:])/10))
        for i in range(int(len(av_dist[0,0,:,:])/10)):
            for l in range(int(len(av_dist[0,0,:,:])/10)):
                KL_value=self.torch_KL(insert_dist[:,:,i*10:i*10+10,l*10:l*10+10]+1
                                    ,av_dist[:,:,i*10:i*10+10,l*10:l*10+10]+1)
                print("{} to {} dist shape is {}".format(i*10,i*10+10,insert_dist[0,0,i*10:i*10+10,l*10:l*10+10].shape))
                print("{} and {}".format(i,l))
                KL_list.append(KL_value)
                # ipdb.set_trace()
        #======================
        
        disappend_index_list=[]
        #その中でKLが高い順で3つとってくる
        KL_array=np.array(KL_list)
        # ipdb.set_trace()
        for i in range(3):
            disappend_index=np.argmax(KL_array)
            KL_array=np.delete(KL_array,disappend_index)
            disappend_index_list.append(disappend_index)
            # ipdb.set_trace()
        # ipdb.set_trace()
        print("No.1 is {}, No.2 is {}, No.3 is {}".format(disappend_index_list[0],disappend_index_list[1]+1,disappend_index_list[2]+2))
        
        No1_info_coordinate=[int(disappend_index_list[0]/10),(int(disappend_index_list[0])%10)-1]
        No2_info_coordinate=[int((int(disappend_index_list[1]+1)/10)),(int(disappend_index_list[1]+1)%10)-1]
        No3_info_coordinate=[int((int(disappend_index_list[2]+2)/10)),(int(disappend_index_list[2]+2)%10)-1]
        
        # ipdb.set_trace()
        return No1_info_coordinate,No2_info_coordinate,No3_info_coordinate
    
    def transform_coordinate(self,no1,no2,no3):
        # -1~1までに変換する必要がある
        # 0~9にしている

        first_x=((no1[0]/9)*2)-1
        second_x=((no2[0]/9)*2)-1
        third_x=((no3[0]/9)*2)-1
        first_y=((no1[1]/9)*2)-1
        second_y=((no2[1]/9)*2)-1
        third_y=((no3[1]/9)*2)-1
        
        return [first_x,first_y],[second_x,second_y],[third_x,third_y]
    
    
if __name__ == "__main__":
    P = torch.Tensor([[0.36, 0.48, 0.16],[0.36, 0.48, 0.16]])
    Q = torch.Tensor([[0.333, 0.333, 0.333],[0.333, 0.333, 0.333]])
    
    
    K=KL_dist()
    D=Data_dist()
    X_b,Y_b,Z_b=D.baseline()
    
    f = open('val_10_ver1.txt', 'r')
    if f.mode=='r':
        contents= f.read()
        Z_av=np.array(eval(contents)['distZ'])
    
    
    f = open('val_10_ver2.txt', 'r')
    if f.mode=='r':
        contents= f.read()
        Z_1=np.array(eval(contents)['distZ'])
    
    
    # print(Z_1[:,0,:,:].shape)
    Z_first_temp=Z_1[1,0,:,:]
    
    Z_first_av=Z_av[:,0,:,:]
    Z_first_1=Z_1[:,0,:,:]
    dist_av=torch.Tensor(Z_first_av)
    dist_1=torch.Tensor(Z_first_1)
    
    # for debag
    av_dist=torch.Tensor(np.stack([Z_b,Z_b,Z_b,Z_b,Z_b,Z_b]))
    dist_sample1=torch.Tensor((np.stack([np.stack([Z_b,Z_b]),np.stack([Z_b,Z_b])])+0.11)*4)
    dist_sample2=torch.Tensor((np.stack([np.stack([Z_b,Z_b]),np.stack([Z_b,Z_b])])+0.15)*1.4)
    # ipdb.set_trace()
    
    # KL_value=K.torch_KL(dist_sample1,dist_sample2)
    KL_value=K.torch_KL(torch.tensor(Z_av[:,:,:,:]+1),torch.tensor(Z_1[:,:,:,:]+1))
    index_list_1,index_list_2,index_list_3,space_KL=K.sprit_calcu_time(torch.tensor(Z_av[:,:,:,:]),torch.tensor(Z_1[:,:,:,:]),0)
    #全時刻全センサー種から計算されたKL距離から目標位置3点を決定している
    No1_info_coordinate,No2_info_coordinate,No3_info_coordinate=K.sprit_calcu_all(torch.tensor(Z_av[:,:,:,:]),torch.tensor(Z_1[:,:,:,:]))
    #ここで値を変換している
    no1,no2,no3=K.transform_coordinate(No1_info_coordinate,No2_info_coordinate,No3_info_coordinate)
    
    print("Coordinate ___ No.1 is {}. No.2 is {}. No.3 is {}".format(No1_info_coordinate,No2_info_coordinate,No3_info_coordinate))
    
    
    
    #======================
    #--------------
    #テスト時に軌道を受け取って各位置の情報量を吐き出す処理
    #--------------
    x=y=[0,0,0,0,0]
    position_KL_list=[]
    
    for i in range(len(x)):
        position_KL=K.position_KL_calcu(space_KL,x[i],y[i])
        position_KL_list=np.append(position_KL_list,position_KL)
        # ipdb.set_trace()
        print(i)
    #======================
    
    print("{} is KL value".format(abs(KL_value)))
    
    # making graph
    # ax = Axes3D(plt.figure())
    # ax.plot_wireframe(X_b, Y_b, Z_first_temp)
    ipdb.set_trace()
    # plt.show()

