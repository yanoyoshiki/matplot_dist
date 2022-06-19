import torch
import torch.nn.functional as F

class KL_dist():
    def torch_KL(self,first_dist,second_dist):
        torch_KL_d_value=F.kl_div(first_dist.log(), second_dist, None, None, 'sum')
        return torch_KL_d_value

if __name__ == "__main__":
    P = torch.Tensor([[0.36, 0.48, 0.16],[0.36, 0.48, 0.16]])
    Q = torch.Tensor([[0.333, 0.333, 0.333],[0.333, 0.333, 0.333]])
    
    
    K=KL_dist()
    KL_value=K.torch_KL(P,Q)
    print("{} is KL value".format(KL_value))
