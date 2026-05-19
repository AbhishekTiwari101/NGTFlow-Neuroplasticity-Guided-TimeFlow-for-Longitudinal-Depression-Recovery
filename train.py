import torch
from ngtflow.model import NGTFlow
from ngtflow.data import LongitudinalMRIDataset
import yaml

def main():
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    model = NGTFlow().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    dataset = LongitudinalMRIDataset(data_dir="data/LDMD-2025")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    for epoch in range(100):
        for batch in loader:
            x0, t, x_target, madrs = batch
            x0, t = x0.cuda(), t.cuda()
            
            optimizer.zero_grad()
            x_pred, madrs_pred, _ = model(x0, t)
            loss = ngtflow_loss(x_pred, x_target.cuda(), madrs_pred, madrs.cuda())
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
