import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances(tensor_a: torch.Tensor, tensor_b: torch.Tensor, norm_p=2):
    if len(tensor_a.shape) != 3:
        raise ValueError("Expected 3D tensor for tensor_a, got shape: {}".format(tensor_a.shape))
    if len(tensor_b.shape) != 3:
        raise ValueError("Expected 3D tensor for tensor_b, got shape: {}".format(tensor_b.shape))
        
    return (tensor_a.unsqueeze(2) - tensor_b.unsqueeze(1)).abs().pow(norm_p).sum(3)

def chamfer_loss(point_set_a, point_set_b):
    distance_matrix = pairwise_distances(point_set_a, point_set_b)
    dist_a_to_b = torch.mean(torch.sqrt(distance_matrix.min(1)[0]))
    dist_b_to_a = torch.mean(torch.sqrt(distance_matrix.min(2)[0]))
    return dist_a_to_b + dist_b_to_a

def chamfer_distance(predicted_points, target_points):
    return chamfer_loss(predicted_points, target_points)

class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, predicted_points, target_points):
        return chamfer_distance(predicted_points, target_points)

def hausdorff_loss(point_set_a, point_set_b):

    distance_matrix = pairwise_distances(point_set_a, point_set_b)
    max_min_a_to_b = torch.max(torch.min(distance_matrix, dim=2)[0])
    max_min_b_to_a = torch.max(torch.min(distance_matrix, dim=1)[0])
    return torch.max(max_min_a_to_b, max_min_b_to_a)

def hausdorff_distance(predicted_points, target_points):

    return hausdorff_loss(predicted_points, target_points)

class HausdorffDistanceLoss(nn.Module):    
    def __init__(self):
        super(HausdorffDistanceLoss, self).__init__()
    
    def forward(self, predicted_points, target_points):
        return hausdorff_distance(predicted_points, target_points)