import collections
import torch
from tqdm import tqdm
from utils.Evaluation_metrics import Metrics
from model.model import Model
import numpy as np
import random
import warnings
from torch_geometric.data import DataLoader


from sklearn.model_selection import train_test_split

def Eval(model, val_dataset, device):
    model.eval()
    epoch_loss_valid = 0.0
    test_pred = []
    test_label = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataset)):
            pred = model(batch, device)
            label = batch.y
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
            batch_loss = model.criterion(pred, label.to(torch.float))
            epoch_loss_valid += batch_loss.item()
        loss = epoch_loss_valid / len(val_dataset)
        metrics = Metrics(torch.tensor(test_pred), torch.tensor(test_label))
    return loss, metrics,

def train(model, train_dataset, val_dataset,opt, epoch_num, device):
    best_score = float('inf')
    min_delta = 0.00001
    patience = 10
    counter = 0
    early_stop = False
    path = 'state_dict_model.pth'
    for epoch in range(epoch_num):
        model.train()
        print('第{}个epoch*************************************************'.format(epoch))
        epoch_loss_train = 0.0
        for i, batch in tqdm(enumerate(train_dataset)):
            torch.cuda.empty_cache()
            batch = batch.to(device)
            pred = model(batch, device)
            label = batch.y
            batch_loss = model.criterion(pred, label.to(torch.float))
            opt.zero_grad()
            batch_loss.requires_grad_(True)
            batch_loss.backward()
            opt.step()
            epoch_loss_train += batch_loss.item()

        train_loss = epoch_loss_train / len(train_dataset)
        val_loss,  metrics = Eval(model, val_dataset, device)
        print('train_loss  {}'.format(train_loss))
        print('val_loss  {}'.format(val_loss))
        print('ACC :   {}'.format('%.4f' % metrics[0]))
        print('AP :   {}'.format('%.4f' % metrics[1]))
        print('HL :   {}'.format('%.4f' % metrics[2]))
        print('F1 :   {}'.format('%.4f' % metrics[3]))
        print('jaccard :   {}'.format('%.4f' % metrics[4]))
        if best_score == None:
            best_score = metrics[3]
            torch.save(model.state_dict(), path)
        elif best_score - metrics[3] >= min_delta:
            best_score = metrics[3]
            counter = 0
            torch.save(model.state_dict(), path)
        elif best_score - metrics[3] < min_delta:
            counter += 1
            print(f"INFO: Early stopping counter {counter} of {patience}")
            if counter >= patience:
                print('INFO: Early stopping')
                early_stop = True

        if early_stop:
            print("Early stopping")
            break  



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data_list = torch.load('./data/traindata_x.pt')
    test_data_list = torch.load('./data/valdata_x.pt')
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    epoch_num = 1000
    model = model.to(device)
    train_loader = DataLoader(train_data_list, shuffle=True, batch_size=1, drop_last=True)
    val_loader = DataLoader(test_data_list, shuffle=True, batch_size=1, drop_last=True)

    train(model, train_loader, val_loader, opt, epoch_num, device, )


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    set_seed(1234)
    main()

    val_data_list = torch.load('./data/valdata_x.pt')
    val_loader = DataLoader(val_data_list, shuffle=False, batch_size=1, drop_last=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    model.load_state_dict(torch.load('state_dict_model.pth'))
    loss,  metrics = Eval(model, test_loader, device)
    print('ACC :   {}'.format('%.4f' % metrics2[0]))
    print('AP :   {}'.format('%.4f' % metrics2[1]))
    print('HL :   {}'.format('%.4f' % metrics2[2]))
    print('F1 :   {}'.format('%.4f' % metrics2[3]))
    print('jaccard :   {}'.format('%.4f' % metrics2[4]))


