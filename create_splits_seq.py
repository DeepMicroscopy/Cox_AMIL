import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=[
    'task_1_tumor_vs_normal', 
    'task_2_tumor_subtyping', 
    'task_3_survival_prediction',
    'task_3_survival_prediction_augmented',
    'task_3_survival_prediction_after_T', 
    'task_4_tumor_grading_kat2',
    'task_4_tumor_grading_kat4', 
    'task_5_tumor_subtyping',
    'task_6_survival_prediction_augmented',
    'task_7_tumor_grading_kat2_augmented',
    'task_7_tumor_grading_kat4_augmented',
    'task_8_tumor_subtyping_augmented',
    'task_9_survival_prediction_augmented_random'])
parser.add_argument('--csv_path', type=str, default=None, help='Path to csv dataset.')
parser.add_argument('--split_name', type=str, default=None, help='Name of split folder.')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_3_survival_prediction':
    args.n_classes=2

    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path

    dataset = Generic_WSI_Survival_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'lebt':0, 'tod':1},
                        event_col = 'event',
                        time_col = 'time',
                        patient_strat=True,
                        ignore=[])


elif args.task == 'task_3_survival_prediction_augmented':
    args.n_classes=2

    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path

    dataset = Generic_WSI_Survival_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'lebt':0, 'tod':1},
                        event_col = 'event',
                        time_col = 'time',
                        patient_strat=True,
                        ignore=[])

elif args.task == 'task_3_survival_prediction_after_T':
    args.n_classes=2

    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path

    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'lebt':0, 'tod':1},
                        label_col = 'Survival_after_T',
                        patient_strat=True,
                        ignore=[])

elif args.task == 'task_4_tumor_grading_kat2':
    args.n_classes=2
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'G1 G2':0, 'G3 G4':1},
                        label_col = 'Grading_kat2',
                        patient_strat=True,
                        ignore=[])


elif args.task == 'task_4_tumor_grading_kat4':
    args.n_classes=4
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'niedriger Malignitätsgrad':0, 'mittlerer Malignitätsgrad':1, 'hoher Malignitätsgrad':2, 'sehr hoher Malignitätsgrad':3},
                        label_col = 'Grading_kat4',
                        patient_strat=True,
                        ignore=[])


elif args.task == 'task_5_tumor_subtyping':
    args.n_classes=4
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'Plattenepithelkarzinom':0, 'Adenokarzinom+BAC':1, 'grosszelliges Karzinom':2, 'NSCLC NOS':3},
                        label_col = 'Histo_kat6',
                        patient_strat=True,
                        ignore=[])


elif args.task == 'task_6_survival_prediction_augmented':
    args.n_classes=2
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'lebt':0, 'tod':1},
                        label_col = 'Survival_after_T',
                        patient_strat=True,
                        ignore=[])

elif args.task == 'task_7_tumor_grading_kat2_augmented':
    args.n_classes=2
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'G1 G2':0, 'G3 G4':1},
                        label_col = 'Grading_kat2',
                        patient_strat=True,
                        ignore=[])

elif args.task == 'task_7_tumor_grading_kat4_augmented':
    args.n_classes=4
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'niedriger Malignitätsgrad':0, 'mittlerer Malignitätsgrad':1, 'hoher Malignitätsgrad':2, 'sehr hoher Malignitätsgrad':3},
                        label_col = 'Grading_kat4',
                        patient_strat=True,
                        ignore=[])

elif args.task == 'task_8_tumor_subtyping_augmented':
    args.n_classes=4
    if args.csv_path == None:
        raise ValueError('Must provide a csv dataset file.')
    else:
        csv_path = args.csv_path
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'Plattenepithelkarzinom':0, 'Adenokarzinom+BAC':1, 'grosszelliges Karzinom':2, 'NSCLC NOS':3},
                        label_col = 'Histo_kat6',
                        patient_strat=True,
                        ignore=[])
elif args.task == 'task_9_survival_prediction_augmented_random':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ammeling/projects/TMA/annotations/aug_survival_prediction_random.csv',
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = {'lebt':0, 'tod':1},
                        label_col = 'Survival_Status',
                        patient_strat=True,
                        ignore=[])


else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    if args.split_name is not None:
        split_name = args.split_name
    else:
        split_name = ''
        
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task)  +'_{}_{}'.format(split_name, int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



