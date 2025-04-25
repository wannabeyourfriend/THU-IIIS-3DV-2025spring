import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    """
    Compute the pairwise distance_tensor matrix between a and b which both have size [m, n, d]. The result is a tensor of
    size [m, n, n] whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", b.shape)
    return (a.unsqueeze(2) - b.unsqueeze(1)).abs().pow(p).sum(3)

def chamfer(a, b):
    """
    Compute the chamfer distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    M = pairwise_distances(a, b)
    dist1 = torch.mean(torch.sqrt(M.min(1)[0]))
    dist2 = torch.mean(torch.sqrt(M.min(2)[0]))
    return (dist1 + dist2) 


def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
	try:
		from .cuda.chamfer_distance import ChamferDistance
		cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
		cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
		cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
		# chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
		chamfer_loss = (cost_p0_p1 + cost_p1_p0)
  
	except:
		chamfer_loss = chamfer(template, source)
	return chamfer_loss


class ChamferDistanceLoss(nn.Module):
	def __init__(self):
		super(ChamferDistanceLoss, self).__init__()

	def forward(self, template, source):
		return chamfer_distance(template, source)


def hausdorff(a, b):
    """
    Compute the Hausdorff distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Hausdorff distance between each minibatch entry
    """
    M = pairwise_distances(a, b)
    # 计算 d(P,Q) = max_{p_i ∈ P} [min_{q_j ∈ Q} d(p_i, q_j)]
    d_P_Q = torch.max(torch.min(M, dim=2)[0])
    # 计算 d(Q,P) = max_{q_j ∈ Q} [min_{p_i ∈ P} d(q_j, p_i)]
    d_Q_P = torch.max(torch.min(M, dim=1)[0])
    # 取两者的最大值
    return torch.max(d_P_Q, d_Q_P)

def hausdorff_distance(template: torch.Tensor, source: torch.Tensor):
    """
    Compute the Hausdorff distance between template and source point clouds
    :param template: Template point cloud tensor of shape [m, n_a, d]
    :param source: Source point cloud tensor of shape [m, n_b, d]
    :return: Hausdorff distance between template and source
    """
    return hausdorff(template, source)

class HausdorffDistanceLoss(nn.Module):
    def __init__(self):
        super(HausdorffDistanceLoss, self).__init__()
    
    def forward(self, template, source):
        return hausdorff_distance(template, source)