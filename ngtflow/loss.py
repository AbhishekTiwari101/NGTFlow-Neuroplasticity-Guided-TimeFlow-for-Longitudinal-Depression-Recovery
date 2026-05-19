import torch
import torch.nn.functional as F

def ngtflow_loss(x_pred, x_target, madrs_pred, madrs_target, 
                v, hippo_prior, lambda_hippo=0.4):
    
    # Similarity loss
    sim_loss = 1 - F.cosine_similarity(x_pred.flatten(), x_target.flatten(), dim=0)
    
    # Flow regularization (smoothness)
    flow_loss = torch.mean(torch.abs(torch.div(torch.diff(v, dim=0), 1e-5)))  # divergence/curl approx
    
    # Hippocampal constraint
    hippo_loss = F.mse_loss(hippo_prior, torch.ones_like(hippo_prior) * 0.85)  # example prior
    
    # Symptom regression
    sym_loss = F.mse_loss(madrs_pred, madrs_target)
    
    total_loss = (sim_loss + 0.1 * flow_loss + 
                  lambda_hippo * hippo_loss + 0.5 * sym_loss)
    return total_loss
