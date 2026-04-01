import sys
import torch


def compute_joint(view1, view2, EPS=1e-8):
    """Compute the joint probability matrix P计算两个视图的联合概率矩阵 """

    #bn, k = view1.size()
    #assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.

    sum_p_i_j = p_i_j.sum()
    #print(sum_p_i_j)


    p_i_j = p_i_j / (p_i_j.sum() + EPS)

    return p_i_j


def MMI(view1, view2, lamb=1.0, EPS=1e-8):
    """MMI loss用于计算最大互信息（MMI）损失"""
    #check_for_nan(view1, "view1")
    #check_for_nan(view2, "view2")
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2, EPS)



    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

   # p_i_j[(p_i_j < EPS).data] = EPS
   # p_j[(p_j < EPS).data] = EPS
  #  p_i[(p_i < EPS).data] = EPS

    p_i_j = torch.clamp(p_i_j, min=EPS)
    p_j = torch.clamp(p_j, min=EPS)
    p_i = torch.clamp(p_i, min=EPS)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb) * torch.log(p_j) \
                      - (lamb) * torch.log(p_i))

    loss = loss.sum()

    return loss


def check_for_nan(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {tensor_name} contains NaN values.")
    else:
        print(f"{tensor_name} does not contain NaN values.")