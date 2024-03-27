import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]


        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])


        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index
    

class Dataset_vitaldb(Dataset):
    def __init__(self,mode="train", **kwargs):
        # 여기서 엣지 만들고 
        # 데이터 만들기
            # 레이블 어떻게 줄지 생각해야함 
        self.mode = mode
        self.step = 1
        self.win_size = 10
        # self.data_path = data_path
        # get_fc_graph_struc
        struc_map = {}
        feature_map = [str(i) for i in range(4)]
        # feature_list = []
        feature_list = feature_map.copy()

        for ft in feature_list:
            if ft not in struc_map:
                struc_map[ft] = []

            for other_ft in feature_list:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        fc_struc = struc_map

        self.edge_index = build_loc_net(fc_struc, feature_map, feature_map)
        self.edge_index = torch.tensor(self.edge_index, dtype = torch.long)


        self.__read_data__()

    def __read_data__(self):
        import sys
        import yaml
        from random import randint
        import pandas as pd
        sys.path.append('/home/mjh319/workspace/3_hypotension_detection/3_hypo')
        sys.path.append('/home/mjh319/workspace/3_hypotension_detection/3_hypo/models')
        from datasets.dataset import Make_hypo_dataset

        config_path = '/home/mjh319/workspace/_hypo/4_hypo/config/0916_time.yml'
        opt = yaml.load(open(config_path), Loader=yaml.FullLoader)
        # opt.update(vars(self.args))

        dfcases = pd.read_csv("https://api.vitaldb.net/cases")

        opt['invasive'] = True
        opt['multi'] = True
        opt['pred_lag'] = 300
        opt['features'] = 'none'

        # random_key = randint(0, 100000)
        random_key = 777

        loader = Make_hypo_dataset(opt, random_key,dfcases)
        self.dataset_list_train = loader["train"].dataset.dataset_list_
        self.transform_train = loader["train"].dataset.transform
        self.dataset_list_test = loader["test"].dataset.dataset_list_
        self.transform_test = loader["test"].dataset.transform
        self.dataset_list_val = loader["valid"].dataset.dataset_list_
        self.transform_val = loader["valid"].dataset.transform
        self.feature_keys = loader["train"].dataset.feature_keys

        downsample_factor = 30  # 예시로 10개의 데이터 포인트를 평균화하여 다운샘플링
        

        for key in self.feature_keys:
            # print(key)
            tp = self.dataset_list_train[key]
            downsampled_time_series = []
            targets = [] 
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                # if self.dataset_list_train['target'][i] == 0:
                downsampled_time_series.append(downsampled_segment)
                    # targets.append(self.dataset_list_train['target'][i] )

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_train[key] = downsampled_time_series
        # targets = np.stack(targets)
        # self.dataset_list_train['target'] = targets

        
        for key in self.feature_keys:
            # print(key)
            tp = self.dataset_list_test[key]
            downsampled_time_series = []
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                downsampled_time_series.append(downsampled_segment)

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_test[key] = downsampled_time_series

        for key in self.feature_keys:
            tp = self.dataset_list_val[key]
            downsampled_time_series = []
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                downsampled_time_series.append(downsampled_segment)

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_val[key] = downsampled_time_series


    def __getitem__(self, index):
        edge_index = self.edge_index.long()
        if self.mode == "train":
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_train[feature](np.expand_dims(self.dataset_list_train[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_train[feature](np.expand_dims(self.dataset_list_train[feature][index,:], axis=0))], dim=0) 
            return data_.squeeze().double(),\
                    data_.squeeze().double()[:,-1],\
                  torch.tensor(self.dataset_list_train['target'][index]),\
                  edge_index
                  
        elif (self.mode == 'val'):
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_val[feature](np.expand_dims(self.dataset_list_val[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_val[feature](np.expand_dims(self.dataset_list_val[feature][index,:], axis=0))], dim=0) 
            return data_.squeeze().double(),\
                    data_.squeeze().double()[:,-1],\
                  torch.tensor(self.dataset_list_val['target'][index]),\
                  edge_index
        elif (self.mode == 'test'):
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))], dim=0) 
            return data_.squeeze().double(),\
                    data_.squeeze().double()[:,-1],\
                  torch.tensor(self.dataset_list_test['target'][index]),\
                  edge_index
        else:
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))], dim=0) 
            return np.array(data_.squeeze().transpose(1,0).to(torch.float32)), np.array(np.float32(self.dataset_list_test['target'][index]))


    def __len__(self):
        if self.mode == "train":
            return len(self.dataset_list_train['target'])
        elif (self.mode == 'val'):
            return len(self.dataset_list_val['target'])
        elif (self.mode == 'test'):
            return len(self.dataset_list_test['target'])
        else:
            return len(self.dataset_list_test['target'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





