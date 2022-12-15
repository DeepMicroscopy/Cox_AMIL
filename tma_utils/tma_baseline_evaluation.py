"""

Adapted from https://github.com/luisvalesilva/multisurv/blob/master/src/evaluation.py


"""


import numpy as np
import pandas as pd
import torch
import warnings
import torchtuples as tt

from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn_pandas import DataFrameMapper 


class _BaseEvaluationData:
    def __init__(self, model, dataset):
        self._cols_standardize = ['Alter_bei_ED']
        self._cols_targets = ['time', 'event']
        self._cols_leave = np.setdiff1d(dataset.columns, self._cols_standardize + self._cols_targets).tolist()
        
        self.model = model 
        self.data = dataset
        self.type = self._model_type()

        if 'DataFrame' not in str(type(self.data)):
            raise ValueError(f'{self.type} model requires data' +
                                ' in pandas DataFrame.')

        if self.type == 'pycox':
            self.x, self.y = self._process_for_pycox()

    def _model_type(self):
        if 'lifelines' in str(type(self.model)):
            model_type = 'lifelines'
        elif 'pysurvival' in str(type(self.model)):
            model_type = 'pysurvival'
        elif 'pycox' in str(type(self.model)):
            model_type = 'pycox'
        else:
           raise ValueError('"model" is not recognized.')

        return model_type

    def _process_for_cph(self):
        def _get_data(df: pd.DataFrame) -> np.ndarray:
            """Creates dummies for the categorical variables."""
            for col in self._cols_categorical:
                one_hot = pd.get_dummies(df[col], prefix=col)
                df = df.drop(col, axis=1)
                df = df.join(one_hot)
            return df
        return _get_data(self.data)


    def _process_for_pycox(self):
        standardize = [([col], StandardScaler()) for col in self._cols_standardize]
        leave = [(col, None) for col in self._cols_leave]
        
        x_mapper_float = DataFrameMapper(standardize + leave)

        x_transform = lambda df: tt.tuplefy(
            x_mapper_float.fit_transform(df).astype('float32'), 
            )

        get_target = lambda df: (
            df['time'].values.astype('float32'),
            df['event'].values.astype('float32')
            )

        x = x_transform(self.data)
        y = get_target(self.data)

        return x, y


class _BaseEvaluation(_BaseEvaluationData):
    """Evaluation functionality common to all baseline model types."""
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

        self.patient_predictions = None
        self.c_index = None
        self.c_index_td = None
        self.ibs = None
        self.inbll = None

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


class _BaselineModelEvaluation(_BaseEvaluation):
    """Evaluation of baseline models."""
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def _collect_patient_ids(self):
        return self.data.index

    def _predict_risk(self, patient):
        patient_data = self.data.loc[patient]
        time, event = patient_data['time'], patient_data['event']
        data = patient_data.drop(['time', 'event'], axis=1)

        if self.type == 'lifelines':
            pred = self.model.predict_partial_hazard(data)[0]
        elif self.type == 'pysurvival':
            pred = self.model.predict_risk(data)[0]
        else:
            raise NotImplementedError

        return time, event, pred

    def _predict_risk_data(self):
        time, event = self.data['time'], self.data['event']
        data = self.data.drop(['time', 'event'], axis=1)

        if self.type == 'lifelines':
            pred = self.model.predict_partial_hazard(data)
        elif self.type == 'pysurvival':
            pred = self.model.predict_risk(data)
        elif self.type == 'pycox':
            pred = self.model.predict(self.x, batch_size=32, eval_=True, to_cpu=True, num_workers=4)
        else:
            raise NotImplementedError

        return time, event, pred

    def _predict(self):
        data = self.data.drop(['time', 'event'], axis=1)

        if self.type == 'lifelines':
            pred = self.model.predict_survival_function(data)
            pred_times = pred.index
        elif self.type == 'pysurvival':
            pred = self.model.predict_survival(data)
            pred_times = self.model.times
            pred = pd.DataFrame(pred.transpose())
        elif self.type == 'pycox':
            pred = self.model.predict_surv_df(self.x)
            pred_times = pred.index
        else:
            raise NotImplementedError

        times, events = self.data['time'], self.data['event']

        # return times, events, torch.from_numpy(interp_pred)
        return times, events, torch.from_numpy(pred.values)



class Evaluation(_BaseEvaluation):
    """Functionality to compute model evaluation metrics.
    Parameters
    ----------
    model: PyTorch or baseline model
        Model to predict patient survival.
    dataset: pandas.core.frame.DataFrame Dataset of patients..
    """
    def __init__(self, model, dataset):
        super().__init__(model, dataset)


        # TODO: add _CLAM_ModelEvaluation here 

        self.ev = _BaselineModelEvaluation(model, dataset)

    def _collect_patient_predictions(self):
        # Get all patient labels and predictions
        patient_data = {}

        pids = self.ev._collect_patient_ids()

        print(f'Collect patient predictions...', end='')
        times, events, pred = self.ev._predict()

        for i, patient in enumerate(pids): 
            patient_data[patient] = {'time': times.values[i],
                                        'event': events.values[i],
                                        'probabilities': pred[:, i]}

        print()
        print()

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

    def compute_metrics(self, time_points=None):
        """Calculate evaluation metrics."""
        if self.patient_predictions is None:
            # Get all patient labels and predictions
            self.patient_predictions = self._collect_patient_predictions()

        if self.c_index is None:
            self.c_index = self._compute_c_index(self.patient_predictions)

        if self.c_index_td is None:
            td_metrics = self._compute_pycox_metrics(self.patient_predictions,
                                                     time_points)
            self.c_index_td, self.ibs, self.inbll = td_metrics

    def run_bootstrap(self, n=1000, time_points=None):
        """Calculate bootstrapped metrics.
        Parameters
        ----------
        n: int
            Number of boostrapped samples.
        time_points: torch.Tensor
            Time points of the predictions.
        Returns
        -------
        Metrics calculated on original dataset and bootstrap samples.
        """
        n = int(n)
        if n <= 0:
            raise ValueError('"n" must be greater than 0.')

        if self.c_index is None:
            try:
                self.compute_metrics(time_points)
            except ZeroDivisionError as err:
                return err, 'C-index could not be calculated.'

        # Run bootstrap
        print('Bootstrap')
        print('-' * 9)

        self.boot_c_index = {}
        self.boot_c_index_td, self.boot_ibs, self.boot_inbll = [], [], []
        skipped = 0

        for i in range(n):
            print('\r' + f'{str((i + 1))}/{n}', end='')
            # Get bootstrap sample (same size as dataset)
            boot_ids = resample(self.ev._collect_patient_ids(), replace=True)
            sample_data = {patient: self.patient_predictions[patient]
                           for patient in boot_ids}

            # When running samples with small number of patients (e.g. some
            # individual cancer types) sometimes there are no admissible pairs
            # to compute the C-index (or other metrics).
            # In those cases continue and print a warning at the end
            try:                                                                
                current_cindex = self._compute_c_index(data=sample_data)
                td_metrics = self._compute_pycox_metrics(
                    sample_data, time_points)
                current_ctd, current_ibs, current_inbll = td_metrics
            except ZeroDivisionError as error:                                  
                err = error                                                     
                skipped += 1                                                    
                continue                                                        

            if not self.boot_c_index:
                for j in range(len(current_cindex)):
                    self.boot_c_index.update({str(j): []})

            for k, x in enumerate(current_cindex):
                self.boot_c_index[str(k)].append(x)

            self.boot_c_index_td.append(current_ctd)
            self.boot_ibs.append(current_ibs)
            self.boot_inbll.append(current_inbll)

        print()

        if skipped > 0:
            warnings.warn(
                f'Skipped {skipped} bootstraps ({err}).')

    def format_results(self, method='percentile'):
        """Calculate bootstrap confidence intervals.
        The empirical bootstrap method uses the distribution of differences
        between the metric calculated on the original dataset and on the
        bootstrap samples. The percentile method uses the distribution of the
        metrics calculated on the bootstrap samples directly.
        Parameters
        ----------
        method: str 
            Bootstrap method to calculate confidence intervals (one of
            "percentile" and "empirical").
        Returns
        -------
        Metrics with 95% bootstrap confidence intervals.
        """
        assert self.c_index is not None, 'Results not available.' + \
                ' Please call "compute_metrics" or "run_bootstrap" first.'

        bootstrap_methods = ['percentile', 'empirical']
        assert method in bootstrap_methods, + \
                '"method" must be one of {bootstrap_methods}.'

        c_index = self.c_index[0]
        c_index = round(c_index , 3)
        ctd = round(self.c_index_td, 3)
        ibs = round(self.ibs, 3)
        inbll = round(self.inbll, 3)

        output = {}

        output['C-index'] = str(c_index)
        output['Ctd'] = str(ctd)
        output['IBS'] = str(ibs)
        output['INBLL'] = str(inbll)

        def _get_differences(metric_value, bootstrap_values):
            differences = [x - metric_value for x in bootstrap_values]

            return sorted(differences) 

        def _get_empirical_percentiles(values, metric_value):
            values = _get_differences(metric_value, values)
            percent = np.percentile(values, [2.5, 97.5])
            low, high = tuple(round(metric_value + x, 3)
                              for x in [percent[0], percent[1]])

            return f'({low}-{high})'

        def _get_percentiles(values):
            percent = np.percentile(values, [2.5, 97.5])
            low, high = round(percent[0], 3), round(percent[1], 3)

            return f'({low}-{high})'

        try:
            if method == 'empirical':
                c_index_percent = _get_empirical_percentiles(
                    self.boot_c_index['0'], self.c_index[0])
                c_index_td_percent = _get_empirical_percentiles(
                    self.boot_c_index_td, self.c_index_td)
                ibs_percent = _get_empirical_percentiles(
                    self.boot_ibs, self.ibs)
                inbll_percent = _get_empirical_percentiles(
                        self.boot_inbll, self.inbll)
            else:
                c_index_percent = _get_percentiles(self.boot_c_index['0'])
                c_index_td_percent = _get_percentiles(self.boot_c_index_td)
                ibs_percent = _get_percentiles(self.boot_ibs)
                inbll_percent = _get_percentiles(self.boot_inbll)

            output['C-index'] += ' ' + str(c_index_percent)
            output['Ctd'] += ' ' + str(c_index_td_percent)
            output['IBS'] += ' ' + str(ibs_percent)
            output['INBLL'] += ' ' + str(inbll_percent)
        except:
            return output

        return output

    def show_results(self, method='percentile'):
        results = self.format_results(method)

        if '(' in results['C-index']:  # bootstrap results available?
            print('          Value (95% CI)')
            print('-' * 29)

        for algo, res in results.items():
            print(algo + ' ' * (10 - len(algo)) + res)


    def return_results(self):
        assert all([
            self.patient_predictions,
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