"""


Adapted from https://github.com/luisvalesilva/multisurv/blob/master/src/baseline_models.py


"""

import numpy as np
import pandas as pd
import torchtuples as tt

from lifelines import CoxPHFitter
from pycox.models import CoxPH
from pycox.models import CoxTime
from pycox.models import DeepHitSingle
from pycox.models import LogisticHazard
from pycox.models import MTLR
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from pycox.models.cox_time import MixedInputMLPCoxTime, MLPVanillaCoxTime
from pysurvival.models.survival_forest import RandomSurvivalForestModel

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 


class _BaseData:
    def __init__(self, algorithm, data):
        # # check data 
        # self._cols_standardize = ['Alter_bei_ED']
        # self._cols_leave = ['Sex', 'Rauchen_kat', 'Ki_67_nukleär_kat']
        # self._cols_categorical = ['Grading_kat4', 'Staging']
        # self._cols_targets = ['time', 'event']
        
        # cols = self._cols_standardize \
        #         + self._cols_leave \
        #         + self._cols_categorical \
        #         + self._cols_targets
        
        # for group in data:
        #     for col in cols:
        #         if col not in data[group].columns:
        #             raise ValueError(f'{col} is missing in {group} data.')

        # check data 
        self._cols_standardize = ['Alter_bei_ED']
        self._cols_targets = ['time', 'event']
        self._cols_leave = np.setdiff1d(data['train'].columns, self._cols_standardize + self._cols_targets).tolist()
        
        # cols = self._cols_standardize \
        #         + self._cols_leave \
        #         + self._cols_targets
        
        # for group in data:
        #     for col in cols:
        #         if col not in data[group].columns:
        #             raise ValueError(f'{col} is missing in {group} data.')

        # check methods 
        self._lifelines_methods = ['CPH']
        self._pysurvival_methods = ['RSF']
        self._pycox_methods = ['DeepSurv', 'CoxTime', 'DeepHIT', 'Nnet-survival']
        self._discrete_time_methods = ['DeepHIT', 'Nnet-survival']

        methods = self._lifelines_methods \
                    + self._pysurvival_methods \
                    + self._pycox_methods

        if not algorithm in methods:
            raise ValueError(f'{algorithm} is not a recognized algorithm.')

        self.data = data 
        self.algorithm = algorithm

        # if self.algorithm in self._lifelines_methods:
        #     self.data = self._process_for_cph()

        if self.algorithm in self._pycox_methods:
            self.x, self.y, self.val = self._process_for_pycox()

    def _process_for_cph(self):
        def _get_data(df: pd.DataFrame) -> np.ndarray:
            """Creates dummies for the categorical variables."""
            for col in self._cols_categorical:
                one_hot = pd.get_dummies(df[col], prefix=col)
                df = df.drop(col, axis=1)
                df = df.join(one_hot)
            return df
        
        data = {group: _get_data(self.data[group]) for group in self.data}

        return data


    def _process_for_pycox(self):

        standardize = [([col], StandardScaler()) for col in self._cols_standardize]
        leave = [(col, None) for col in self._cols_leave]
        
        x_mapper_float = DataFrameMapper(standardize + leave)

        x_fit_transform = lambda df: tt.tuplefy(
            x_mapper_float.fit_transform(df).astype('float32'), 
            )
        x_transform = lambda df: tt.tuplefy(
            x_mapper_float.transform(df).astype('float32'), 
            )

        get_target = lambda df: (
            df['time'].values.astype('float32'),
            df['event'].values.astype('float32')
            )

        x_train = x_fit_transform(self.data['train'])
        x_val = x_transform(self.data['val'])
        x_test = x_transform(self.data['test'])

        x = {'train': x_train, 'val': x_val, 'test': x_test}
        y = {group: get_target(self.data[group]) for group in self.data}
        val = tt.tuplefy(x['val'], y['val'])

        return x, y, val        

    # def _process_for_pycox(self):

    #     standardize = [([col], StandardScaler()) for col in self._cols_standardize]
    #     leave = [(col, None) for col in self._cols_leave]
    #     categorical = [(col, OrderedCategoricalLong()) for col in self._cols_categorical]
        
    #     x_mapper_float = DataFrameMapper(standardize + leave)
    #     x_mapper_long = DataFrameMapper(categorical) 

    #     x_fit_transform = lambda df: tt.tuplefy(
    #         x_mapper_float.fit_transform(df).astype('float32'), 
    #         x_mapper_long.fit_transform(df).astype('int64')
    #         )
    #     x_transform = lambda df: tt.tuplefy(
    #         x_mapper_float.transform(df).astype('float32'), 
    #         x_mapper_long.transform(df).astype('int64')
    #         )

    #     get_target = lambda df: (
    #         df['time'].values.astype('float32'),
    #         df['event'].values.astype('float32')
    #         )

    #     x_train = x_fit_transform(self.data['train'])
    #     x_val = x_transform(self.data['val'])
    #     x_test = x_transform(self.data['test'])

    #     x = {'train': x_train, 'val': x_val, 'test': x_test}
    #     y = {group: get_target(self.data[group]) for group in self.data}
    #     val = tt.tuplefy(x['val'], y['val'])

    #     return x, y, val


class _BaseModel(_BaseData):
    def __init__(self, algorithm, data):
        super().__init__(algorithm, data)

    def _get_discrete_time_net(self, label_transf, net_args):
        self.y['train'] = label_transf.fit_transform(*self.y['train'])
        self.y['val'] = label_transf.transform(*self.y['val'])
        self.val = (self.x['val'], self.y['val'])

        net = tt.practical.MLPVanilla(
            out_features=label_transf.out_features, **net_args)

        return net

    def _model_factory(self, n_trees=None, n_input_features=None, n_neurons=None, penalizer=0.0):
        if self.algorithm == 'CPH':
            return CoxPHFitter(penalizer=penalizer)
        elif self.algorithm == 'RSF':
            return RandomSurvivalForestModel(num_trees=n_trees)
        elif self.algorithm in self._pycox_methods:
            # create embeddings for categorical variables 
            # num_embeddings = self.x['train'][0].max(0) + 1
            # embedding_dims = num_embeddings // 2

            net_args = {
                'in_features': n_input_features,
                'num_nodes': n_neurons,
                'batch_norm': True,
                'dropout': 0.2,
                # 'embedding_dims': embedding_dims,
                # 'num_embeddings': num_embeddings
            }
            if self.algorithm == 'DeepSurv':
                net = tt.practical.MLPVanilla(
                    out_features=1, output_bias=False, **net_args)
                model = CoxPH(net, tt.optim.Adam)
                return model 

            if self.algorithm == 'CoxTime':
                net = MLPVanillaCoxTime(**net_args)
                model = CoxTime(net, tt.optim.Adam)
                return model 

            if self.algorithm in self._discrete_time_methods:
                num_durations = 10
                print(f'  {num_durations} equidistant time intervals')
            
            if self.algorithm == 'DeepHit':
                labtrans = DeepHitSingle.label_transform(num_durations)
                net = self._get_discrete_time_net(labtrans, net_args)
                model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1,
                                        duration_index=labtrans.cuts)
                return model 

            if self.algorithm == 'Nnet-survival':
                labtrans = LogisticHazard.label_transform(num_durations)
                net = self._get_discrete_time_net(labtrans, net_args)
                model = LogisticHazard(net, tt.optim.Adam(0.01),
                                        duration_index=labtrans.cuts)
                return model 
        else:
            raise Exception('Unrecognized model.')


class Baseline(_BaseModel):
    def __init__(self, 
        algorithm: str, 
        data: dict, 
        n_trees: int = None, 
        n_neurons: int = None,
        penalizer: float = 0.0) -> None:
        """Fit baseline models. 

        Args:
            algorithm (str): Name of algorithm.
            data (dict): Dict of pandas DataFrames with keys 'train', 'val', and 'test'.
            n_trees (int, optional): Number of trees. Defaults to None.
            n_neurons (int, optional): Number of neurons. Defaults to None.
            penalizer (float, optional): Penalty for coefficients during regression. Defaults to 0.0.
        """
        super().__init__(algorithm, data)
        model_factory_args = {}

        if self.algorithm == 'CPH':
            model_factory_args['penalizer'] = penalizer
        if self.algorithm == 'RSF':
            model_factory_args['n_trees'] = n_trees
        elif self.algorithm in self._pycox_methods:
            model_factory_args['n_input_features'] = self.x['train'][0].shape[1]
            model_factory_args['n_neurons'] = n_neurons

        self.model = self._model_factory(**model_factory_args)

    def fit(self, **kwargs):
        if self.algorithm == 'CPH':
            self.model.fit(
                self.data['train'], duration_col='time', event_col='event',
                **kwargs 
            )
        elif self.algorithm == 'RSF':
            self.model.fit(
                self.data['train'].drop(self._cols_targets, axis=1).values.astype('float32'),
                self.data['train']['time'].values.astype('float32'),
                self.data['train']['event'].values.astype('int64')
            )
        elif self.algorithm in self._pycox_methods:
                lrfinder = self.model.lr_finder(
                    self.x['train'], self.y['train'], tolerance=2)

                # Set LR as ~half of highest LR before training loss explosion
                lr = lrfinder.get_best_lr() * 0.4
                if len(str(lr)) > 5:
                    lr = round(lr, 4)
                print('   Learning rate', lr)
                print('   Batch size', kwargs['batch_size'])
                self.model.optimizer.set_lr(lr)
                print()

                if self.algorithm == 'CoxTime':
                    val_data = self.val.repeat(10).cat()
                else:
                    val_data = self.val

                self.training_log = self.model.fit(
                    self.x['train'], self.y['train'], epochs=200,
                    callbacks=[tt.callbacks.EarlyStopping()],
                    val_data=val_data, **kwargs)
                
                if not self.algorithm in self._discrete_time_methods:
                    _ = self.model.compute_baseline_hazards()



            
            




    
