import lifelines 
import numpy as np
import os
import torch

from datasets.dataset_generic import save_splits
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from models.model_amil import AMIL
from models.model_mil import MIL_fc, MIL_fc_mc, MIL_fc_Surv
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from utils.utils import *
from utils.core_utils import Accuracy_Logger, EarlyStopping


def cox_log_rank(risks, events, times):
    risks = risks.cpu().numpy().reshape(-1)
    times = times.cpu().numpy().reshape(-1)
    events = events.data.cpu().numpy()
    median = np.median(risks)
    risks_dichotomize = np.where(risks > median, 1, 0).astype(int)
    idx = risks_dichotomize == 0
    T1 = times[idx]
    T2 = times[~idx]
    E1 = events[idx]
    E2 = events[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred

def c_index_lifelines(risks, events, times):
    risks = risks.cpu().numpy().reshape(-1)
    times = times.cpu().numpy().reshape(-1)
    events = events.data.cpu().numpy()
    label = []
    risk = []
    surv_time = []
    for i in range(len(risks)):
        if not np.isnan(risks[i]):
            label.append(events[i])
            risk.append(risks[i])
            surv_time.append(times[i])
    new_label = np.asarray(label)
    new_risk = np.asarray(risk)
    new_surv = np.asarray(surv_time)
    return concordance_index(new_surv, -new_risk, new_label)


def neg_partial_loglik(preds, events, times):
    batch_size = len(preds)
    risk_set = np.zeros([batch_size, batch_size], dtype=int)
    for i in range(batch_size):
        for j in range(batch_size):
            risk_set[i, j] = times[j] >= times[i]
    risk_set = torch.FloatTensor(risk_set).cuda()
    events = torch.FloatTensor(events).cuda()
    
    theta = preds.reshape(-1)
    exp_theta = torch.exp(theta)

    loss = - torch.mean((theta - torch.log(torch.sum(exp_theta * risk_set, dim=1))) * events)
    return loss 



def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='amil':
        model = AMIL(**model_dict)
    elif args.model_type == 'mil':
        model = MIL_fc_Surv(**model_dict)
    else:    
        raise NotImplementedError

    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, survival=True)
    val_loader = get_split_loader(val_split,  testing = args.testing, survival=True)
    test_loader = get_split_loader(test_split, testing = args.testing, survival=True)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):

        train_loop(epoch, model, train_loader, optimizer, args.n_iters, writer)
        stop = validate(cur, epoch, model, val_loader, early_stopping, writer, args.results_dir)
    
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_loss, val_pvalue, val_c_index = summary(model, val_loader)
    print('Val loss: {:.4f}, c_index: {:.4f}, p_value: {:.3e}'.format(val_loss, val_pvalue, val_c_index))

    results_dict, test_loss, test_pvalue, test_c_index = summary(model, test_loader)
    print('Test loss: {:.4f}, c_index: {:.4f}, p_value: {:.3e}'.format(test_loss, test_c_index, test_pvalue))


    if writer:
        writer.add_scalar('final/val_loss', val_loss, 0)
        writer.add_scalar('final/val_c_index', val_c_index, 0)
        writer.add_scalar('final/test_loss', test_loss, 0)
        writer.add_scalar('final/test_c_index', test_c_index, 0)
        writer.close()
    return results_dict, test_c_index, val_c_index




def train_loop(epoch, model, loader, optimizer, n_iters = 16, writer = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    bag_loss_epoch = []

    preds_epoch = None
    preds_batch = None

    time_batch = []
    event_batch = []
    print('\n')

    iter = 0 
    for batch_idx, (data, event, time) in enumerate(loader):
        data, event = data.to(device), event.to(device)

        # ======================= forward pass ======================= # 
        logit, _, _ = model(data)

        # log survival predictions
        time_numpy = np.squeeze(time.data.numpy())
        event_numpy = np.squeeze(event.cpu().data.numpy())

        time_batch.append(time_numpy)
        event_batch.append(event_numpy)

        if batch_idx == 0:
            preds_epoch = logit
            time_torch = time
            event_torch = event.cpu()

        if iter == 0:
            preds_batch = logit

        else:
            preds_epoch = torch.cat([preds_epoch, logit])
            preds_batch = torch.cat([preds_batch, logit])

            time_torch = torch.cat([time_torch, time])
            event_torch = torch.cat([event_torch, event.cpu()])

        iter += 1

        # compute loss over accumulated samples 
        if iter % n_iters == 0 or batch_idx == len(loader)-1:
            time_batch = np.asarray(time_batch)
            event_batch = np.asarray(event_batch)
            if np.max(event_batch) == 0:
                print('Epoch: {}, iter {}, encounterd no event in batch, skip'.format(epoch, iter))
                preds_batch = None
                time_batch = []
                event_batch = []
                iter = 0
                continue
            
            # compute and log loss 
            bag_loss_batch = neg_partial_loglik(preds_batch, event_batch, time_batch)
            bag_loss_epoch.append(bag_loss_batch.item())

            # regularization
            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum() 

            # total loss 
            loss = bag_loss_batch + 1e-5 * l1_reg

            # ======================= backward pass ======================= # 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()
            preds_batch = None
            time_batch = []
            event_batch = []
            iter = 0


    # calculate loss over epoch
    bag_loss_epoch = np.mean(bag_loss_epoch)  

    # calculate metrics
    pvalue_pred = cox_log_rank(preds_epoch.data, event_torch, time_torch)
    c_index = c_index_lifelines(preds_epoch.data, event_torch, time_torch)

    # print progress
    print('Epoch: {}, train_loss: {:.4f}, c_index: {:.4f}, p_value: {:.3e}'.format(
        epoch, bag_loss_epoch, c_index, pvalue_pred)
        )

    if writer:
        writer.add_scalar('train/loss', bag_loss_epoch, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)



def validate(cur, epoch, model, loader, early_stopping = None, writer = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    bag_loss_epoch = []

    preds_batch = None

    time_batch = []
    event_batch = []
    print('\n')

    for batch_idx, (data, event, time) in enumerate(loader):
        data, event = data.to(device), event.to(device)

        # ======================= forward pass ======================= # 
        with torch.no_grad():
            logit, _ , _ = model(data)

        # log survival predictions
        time_numpy = np.squeeze(time.data.numpy())
        event_numpy = np.squeeze(event.cpu().data.numpy())

        time_batch.append(time_numpy)
        event_batch.append(event_numpy)

        if batch_idx == 0:
            preds_batch = logit

            time_torch = time
            event_torch = event.cpu()

        else:
            preds_batch = torch.cat([preds_batch, logit])

            time_torch = torch.cat([time_torch, time])
            event_torch = torch.cat([event_torch, event.cpu()])


    # compute loss over accumulated samples 
    time_batch = np.asarray(time_batch)
    event_batch = np.asarray(event_batch)
    
    # compute and log loss 
    bag_loss_epoch = neg_partial_loglik(preds_batch, event_batch, time_batch)

    # calculate metrics
    pvalue_pred = cox_log_rank(preds_batch.data, event_torch, time_torch)
    c_index = c_index_lifelines(preds_batch.data, event_torch, time_torch)

    # print progress
    print('Epoch: {}, val_loss: {:.4f}, c_index: {:.4f}, p_value: {:.3e}'.format(
        epoch, bag_loss_epoch, c_index, pvalue_pred)
        )

    if writer:
        writer.add_scalar('val/loss', bag_loss_epoch, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)


    if early_stopping:
        assert results_dir
        early_stopping(epoch, bag_loss_epoch, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False



def summary(model, loader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds_batch = None
    time_batch = []
    event_batch = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, event, time) in enumerate(loader):
        data, event = data.to(device), event.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logit, _, _ = model(data)

        # log survival predictions
        time_numpy = np.squeeze(time.data.numpy())
        event_numpy = np.squeeze(event.cpu().data.numpy())

        time_batch.append(time_numpy)
        event_batch.append(event_numpy)

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'hazard': logit.item(), 'event': event.item(), 'time': time.item()}})

        if batch_idx == 0:
            preds_batch = logit

            time_torch = time
            event_torch = event.cpu()

        else:
            preds_batch = torch.cat([preds_batch, logit])

            time_torch = torch.cat([time_torch, time])
            event_torch = torch.cat([event_torch, event.cpu()])


    # compute loss over accumulated samples 
    time_batch = np.asarray(time_batch)
    event_batch = np.asarray(event_batch)
    
    # compute and log loss 
    bag_loss_epoch = neg_partial_loglik(preds_batch, event_batch, time_batch)

    # calculate metrics
    pvalue_pred = cox_log_rank(preds_batch.data, event_torch, time_torch)
    c_index = c_index_lifelines(preds_batch.data, event_torch, time_torch)

    return patient_results, bag_loss_epoch, pvalue_pred, c_index

    

    





