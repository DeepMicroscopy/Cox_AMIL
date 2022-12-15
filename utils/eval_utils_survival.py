import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from lifelines.utils import concordance_index
from models.model_amil import AMIL
from models.model_mil import MIL_fc_Surv
from pycox.evaluation import EvalSurv
from utils.utils import *


def initiate_model(settings, ckpt_path):
    print('Initialize model ...', end=' ')
    model_dict = {"dropout": settings['drop_out']}
    
    if settings['model_size'] is not None and settings['model_type'] == 'amil':
        model_dict.update({"size_arg": settings['model_size']})
    
    if settings['model_type'] =='amil':
        model = AMIL(**model_dict)
    elif settings['model_type'] == 'mil':
        model = MIL_fc_Surv(**model_dict)
    else:
        raise NotImplementedError

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    print('Done.')

    if settings['print_model_info']:
        print_network(model)  

    return model



class _BaseEvaluationData:
    event_col = 'event'
    time_col = 'time'
    
    def __init__(self, settings):
        print('Initialize data ...', end=' ')
        self.dataset = Generic_MIL_Survival_Dataset(csv_path = settings['csv_path'],
            data_dir= os.path.join(settings['data_root_dir'], settings['feature_dir']),
            shuffle = False, 
            print_info = settings['print_data_info'],
            label_dict = {'lebt':0, 'tod':1},
            event_col = self.event_col,
            time_col = self.time_col,
            patient_strat=True,
            ignore=[])

        self.split_path = '{}/splits_{}.csv'.format(settings['split_dir'], settings['split_idx'])
        print('Done.')

    def _get_split_data(self, split):
        assert split in ['train', 'val', 'test', 'all'], 'Split {} not recognized, must be in [train, val, test, all]'.format(split)
        train, val, test = self.dataset.return_splits(from_id=False, csv_path=self.split_path)
        if split == 'train':
            loader = get_simple_loader(train, survival=True)
        elif split == 'val':
            loader = get_simple_loader(val, survival=True)
        elif split == 'test':
            loader = get_simple_loader(test, survival=True)
        elif split == 'all':
            loader = get_simple_loader(self.dataset, survival=True)
        return loader, loader.dataset.slide_data



class _BaseEvaluationAMIL(_BaseEvaluationData):
    def __init__(self, settings):
        super().__init__(settings)

        # init model
        ckpt_path = os.path.join(settings['models_dir'], 's_{}_checkpoint.pt'.format(settings['split_idx']))
        self.model = initiate_model(settings, ckpt_path)

        self.baseline_hazard = None
        self.baseline_cumulutative_hazard = None

        self.patient_predictions = None
        self.c_index = None
        self.c_index_td = None
        self.ibs = None
        self.inbll = None


    def _compute_risk(self, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        risks = []
        events = []
        times = []
        # print('Collect patient predictions ...', end=' ')
        for batch_idx, (data, event, time) in enumerate(loader):
            with torch.no_grad():
                risk, _ , _ = self.model(data.to(device))
            risks.append(risk.item())
            events.append(event.item())
            times.append(time.item())
        # print('Done.')
        return np.asarray(times), np.asarray(events), np.asarray(risks)
            

    def _compute_baseline_harzards(self):
        """Computes the Breslow esimates from the training data.

        Modified from https://github.com/havakv/pycox/blob/0e9d6f9a1eff88a355ead11f0aa68bfb94647bf8/pycox/models/cox.py#L63        
        """
        loader, dataset = self._get_split_data('train')
        _, _, risk_scores = self._compute_risk(loader)
        return  (dataset 
                .assign(exp_risk=np.exp(risk_scores)) 
                .groupby(dataset.time) 
                .agg({'exp_risk': 'sum', 'event': 'sum'})
                .sort_index(ascending=False) 
                .assign(exp_risk=lambda x: x['exp_risk'].cumsum())  
                .pipe(lambda x: x['event']/x['exp_risk']) 
                .iloc[::-1]
                .rename('baseline_hazards'))

    def _compute_baseline_cumulative_hazards(self):
        """Computes baseline and baseline cumulative hazards and stores as class variable"""
        print('Estimate baseline cumulative hazard ...', end=' ')
        base_hazard = self._compute_baseline_harzards()
        self.baseline_hazard = base_hazard
        self.baseline_cumulutative_hazard = base_hazard.cumsum().rename('baseline_cumulative_hazards')
        print('Done.')


    def _predict_survival_function(self, loader):
        """Predicts survival function for given data loader."""
        if self.baseline_cumulutative_hazard is None:
            self._compute_baseline_cumulative_hazards()

        base_ch = self.baseline_cumulutative_hazard.values.reshape(-1, 1).astype(float)
        times, events, risks = self._compute_risk(loader)
        exp_risk = np.exp(risks).reshape(1, -1)
        surv = np.exp(-base_ch.dot(exp_risk))
        return times, events, torch.from_numpy(surv)

    def _predict_risk(self, loader):
        times, events, risks = self._compute_risk(loader)
        return times, events, risks


    def _collect_patient_ids(self, split):
        loader, dataset = self._get_split_data(split)
        return dataset.index


    def _unpack_data(self, data):
        times = [data[patient]['time'] for patient in data]
        events = [data[patient]['event'] for patient in data]
        predictions = [data[patient]['probabilities'] for patient in data]
        return times, events, predictions

    def _compute_c_index(self, data):
        times, events, predictions = self._unpack_data(data)
        probs_by_interval = torch.stack(predictions).permute(1, 0)
        c_index = [concordance_index(event_times=times,
                                     predicted_scores=interval_probs,
                                     event_observed=events)
                   for interval_probs in probs_by_interval]
        return c_index

    def _predictions_to_pycox(self, data, time_points=None):
        predictions = {k: v['probabilities'] for k, v in data.items()}
        df = pd.DataFrame.from_dict(predictions)
        return df



class EvaluationAMIL(_BaseEvaluationAMIL):
    def __init__(self, settings):
        super().__init__(settings)

        self.split = None

    def _check_split_data(self, split):
        if self.split is None:
            self.split = split
        elif self.split != split:
            self.patient_predictions = None
            self.c_index = None
            self.c_index_td = None
            self.ibs = None
            self.inbll = None

    def _collect_patient_predictions(self, split):
        patient_data = dict()
        loader, _ = self._get_split_data(split)
        pids = self._collect_patient_ids(split)
        times, events, surv = self._predict_survival_function(loader)
        for i, patient in enumerate(pids):
            patient_data[patient] = {'time': times[i],
                                        'event': events[i],
                                        'probabilities': surv[:, i]}
        return patient_data


    def _compute_pycox_metrics(self, data, time_points=None,
                               drop_last_times=0):
        times, events, _ = self._unpack_data(data)
        times, events = np.array(times), np.array(events)
        predictions = self._predictions_to_pycox(data, time_points)

        ev = EvalSurv(predictions, times, events, censor_surv='km')
        # Using "antolini" method instead of "adj_antolini" resulted in Ctd
        # values different from C-index for proportional hazard methods (for
        # CNV data); this is presumably due to the tie handling, since that is
        # what the pycox authors "adjust" (see code comments at:
        # https://github.com/havakv/pycox/blob/6ed3973954789f54453055bbeb85887ded2fb81c/pycox/evaluation/eval_surv.py#L171)
        # c_index_td = ev.concordance_td('antolini')
        c_index_td = ev.concordance_td('adj_antolini')

        # time_grid = np.array(predictions.index)
        # Use 100-point time grid based on data
        time_grid = np.linspace(times.min(), times.max(), 100)
        # Since the score becomes unstable for the highest times, drop the last
        # time points?
        if drop_last_times > 0:
            time_grid = time_grid[:-drop_last_times]
        ibs = ev.integrated_brier_score(time_grid)
        inbll = ev.integrated_nbll(time_grid)

        return c_index_td, ibs, inbll


    def compute_metrics(self, split, time_points=None):
        """Calculate evaluation metrics."""
        self._check_split_data(split)

        print('Compute evaluation metrics ... \n', end =' ')
        if self.patient_predictions is None:
            # Get all patient labels and predictions
            self.patient_predictions = self._collect_patient_predictions(split)

        if self.c_index is None:
            self.c_index = self._compute_c_index(self.patient_predictions)

        if self.c_index_td is None:
            td_metrics = self._compute_pycox_metrics(self.patient_predictions,
                                                     time_points)
            self.c_index_td, self.ibs, self.inbll = td_metrics
        print('Done.')


    def predict_risk(self, split):
        loader, _ = self._get_split_data(split)
        return self._predict_risk(loader)


    def return_results(self):
        assert all([
            self.c_index, 
            self.c_index_td,
            self.ibs,
            self.inbll
        ]),  'Results not available.' + \
                ' Please call "compute_metrics" or "run_bootstrap" first.'

        return (
            self.c_index, 
            self.c_index_td,
            self.ibs,
            self.inbll
            )


    


