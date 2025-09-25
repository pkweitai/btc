import numpy as np, torch
from torch import nn
from sklearn.metrics import mean_squared_error, log_loss
class GRUHead(nn.Module):
    def __init__(self, in_dim, hid=96, num_layers=1):
        super().__init__()
        self.gru=nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=num_layers, batch_first=True)
        self.reg_head=nn.Linear(hid,1); self.clf_head=nn.Linear(hid,1)
    def forward(self,x):
        out,_=self.gru(x); h=out[:,-1,:]
        y_reg=self.reg_head(h).squeeze(-1); y_clf=torch.sigmoid(self.clf_head(h)).squeeze(-1)
        return y_reg, y_clf
def make_sequences(X,y,c,win=60):
    Xv=X.values.astype(np.float32); yv=y.values.astype(np.float32); cv=c.values.astype(np.float32)
    seq_X,seq_y,seq_c=[],[],[]
    for i in range(win,len(Xv)):
        seq_X.append(Xv[i-win:i]); seq_y.append(yv[i]); seq_c.append(cv[i])
    return np.stack(seq_X), np.array(seq_y), np.array(seq_c)
def train_gru(X_tr,y_tr,c_tr,X_va,y_va,c_va,win=60,epochs=14,lr=1e-3,wd=1e-5,device='cpu'):
    trX,trY,trC=make_sequences(X_tr,y_tr,c_tr,win); vaX,vaY,vaC=make_sequences(X_va,y_va,c_va,win)
    in_dim=trX.shape[-1]; model=GRUHead(in_dim=in_dim,hid=96,num_layers=1).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    reg_loss=nn.SmoothL1Loss(); clf_loss=nn.BCELoss()
    trX_t=torch.tensor(trX,device=device); trY_t=torch.tensor(trY,device=device); trC_t=torch.tensor(trC,device=device)
    vaX_t=torch.tensor(vaX,device=device); vaY_t=torch.tensor(vaY,device=device); vaC_t=torch.tensor(vaC,device=device)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        yhat,phat=model(trX_t); loss=reg_loss(yhat,trY_t)+clf_loss(phat,trC_t); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): yv, pv=model(vaX_t)
    rmse=float(np.sqrt(mean_squared_error(vaY_t.cpu(), yv.cpu())))
    logl=float(log_loss(vaC_t.cpu(), pv.cpu().clamp(1e-6,1-1e-6)))
    return model, {'rmse':rmse,'logloss':logl}, win
