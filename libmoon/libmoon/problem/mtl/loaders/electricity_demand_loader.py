import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
#from sktime.utils.load_data import load_from_tsfile
import sktime
from sktime.datasets import load_tsf_to_dataframe
import numpy as np
from libmoon.util.constant import root_name
import os




def load_tsf_data(file_path):
    df, meta_data = load_tsf_to_dataframe(file_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(df)
    df.dropna(inplace=True)  # Handle missing values if any
    
    return df, meta_data


class ElectricityDemandData(torch.utils.data.Dataset):
    def __init__(self, state='QUN', transform=None, sequence_length=96, split='train'):
        #Note: here we consider a single time serie, which is determined by `state`
        #Sequence length to detect anomalies (spikes and drops) is set to 2 days (48hours * 2, because it is half hourly)

        middle_folder_name = os.path.join('libmoon', 'problem', 'mtl','mtl_data','electricity_demand')
        self.path = os.path.join(root_name, middle_folder_name, 'australian_electricity_demand_dataset.tsf')

        df, _ = load_tsf_data(self.path)

        self.state = state
        self.transform = transform
        self.sequence_length = sequence_length

        #self.states = df.index.get_level_values("state").unique().tolist()

        # Filter data for the selected state
        df = df.xs(state, level="state")

        self.timestamps = df.index.get_level_values("timestamp")
        self.data_values_all = df["series_value"].values

        self.classification_type = "anomaly"

        #print(self.timestamps)
        #print(self.data_values)

        full_len = len(self.data_values_all)-self.sequence_length

        self.labels_all = self._generate_labels()
        self.data_values_all = self.data_values_all[0:full_len]

        # anomaly or not for stratification
        labels_any = (self.labels_all[:,0] + self.labels_all[:,1]) > 0

        # train/val/test split: 70/10/20 %
        x_train, x_test, y_train, y_test = train_test_split(self.data_values_all, self.labels_all, test_size=.2, random_state=1, stratify=labels_any)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.125, random_state=1)

        if split == 'train':
            self.data_values = x_train
            self.labels = y_train
        elif split == 'val':
            self.data_values = x_val
            self.labels = y_val
        elif split == 'test':
            self.data_values = x_test
            self.labels = y_test
        

        

    def _generate_labels(self):
        labels = np.zeros((len(self.data_values_all) - self.sequence_length, 2))

        for i in range(len(labels)):
            current_timestamp = self.timestamps[i + self.sequence_length]

            if self.classification_type == "peak_hours":
                hour = current_timestamp.hour
                peak_hours = [6, 7, 8, 9, 10, 17, 18, 19, 20, 21]
                labels[i, 0] = 1 if hour in peak_hours else 0  # Peak demand now
                labels[i, 1] = 1 if (hour + 1) % 24 in peak_hours else 0  # Peak demand next hour

            elif self.classification_type == "seasonal":
                month = current_timestamp.month
                labels[i, 0] = 1 if month in [12, 1, 2] else 0  # Summer vs. Other
                labels[i, 1] = 1 if current_timestamp.weekday() >= 5 else 0  # Weekend vs. Weekday

            elif self.classification_type == "anomaly":
                #prev_val = self.data_values[i + self.sequence_length - 1]
                curr_val = self.data_values_all[i + self.sequence_length]

                # Compute rolling mean and standard deviation (past 48 hours)
                rolling_mean = np.mean(self.data_values_all[i : i + self.sequence_length])
                rolling_std = np.std(self.data_values_all[i : i + self.sequence_length])

                # Define anomaly thresholds (2 standard deviations away)
                lower_bound = rolling_mean - 2 * rolling_std
                upper_bound = rolling_mean + 2 * rolling_std

                labels[i, 0] = 1 if np.any(curr_val < lower_bound) else 0  # Drop anomaly
                labels[i, 1] = 1 if np.any(curr_val > upper_bound) else 0  # Spike anomaly

        return labels

    def __len__(self):
        return len(self.data_values)-self.sequence_length

    def __getitem__(self, idx):

        input_seq = self.data_values_all[idx: idx+self.sequence_length]
        label1, label2 = self.labels_all[idx]

        sample = dict(
            data=torch.tensor(input_seq, dtype=torch.float),
            labels_drp=torch.tensor(label1, dtype=torch.long),
            labels_spk=torch.tensor(label2, dtype=torch.long)
        )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def task_names(self):
        return ['drp', 'spk']

    """
    def split_train_test_val(self, train_prop=0.7, test_prop=0.2, random_state=42):
        labels = np.array([self.labels[i, 0] or self.labels[i, 1] for i in range(len(self))])  # Combine both objectives

        # Separate normal and anomaly indices
        normal_indices = np.where(labels == 0)[0]
        anomaly_indices = np.where(labels == 1)[0]

        # Split normal samples
        normal_train_size = int(train_prop * len(normal_indices))
        normal_test_size = int(test_prop * len(normal_indices) )
        normal_val_size = len(normal_indices) - normal_train_size  - normal_test_size
        train_normal, test_normal, val_normal = random_split(normal_indices, [normal_train_size, normal_test_size, normal_val_size], generator=torch.Generator().manual_seed(random_state))

        # Ensure some anomalies in both train & test
        anomaly_train_size = int(train_prop * len(anomaly_indices))
        anomaly_test_size = int(test_prop * len(anomaly_indices))
        anomaly_val_size = len(anomaly_indices) - anomaly_train_size -anomaly_test_size
        train_anomaly, test_anomaly, val_anomaly = random_split(anomaly_indices, [anomaly_train_size, anomaly_test_size, anomaly_val_size], generator=torch.Generator().manual_seed(random_state))

        # Combine balanced indices
        train_indices = np.concatenate([train_normal, train_anomaly])
        test_indices = np.concatenate([test_normal, test_anomaly])
        val_indices = np.concatenate([val_normal, val_anomaly])

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        np.random.shuffle(val_indices)

        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)

        return train_dataset, test_dataset, val_dataset
    """

import numpy as np

# --- If needed, fix the default state in your class definition to 'QLD' ---
# class ElectricityDemandData(... state='QLD', ...)

def summarize_split(state='QUN', sequence_length=96):
    results = {}
    for split in ['train', 'val', 'test']:
        ds = ElectricityDemandData(state=state, sequence_length=sequence_length, split=split)
        y = ds.labels  # shape [N, 2] -> [:,0]=drp, [:,1]=spk
        n = int(y.shape[0])
        drp_pos = int(y[:, 0].sum())
        spk_pos = int(y[:, 1].sum())
        any_pos = int(((y[:, 0] + y[:, 1]) > 0).sum())
        results[split] = dict(
            n=n,
            drp_pos=drp_pos,
            spk_pos=spk_pos,
            any_pos=any_pos
        )
        print(f"[{state}] {split}: N={n}, drp+= {drp_pos}, spk+= {spk_pos}, any+= {any_pos}")
    # aggregate
    N_total = sum(results[s]['n'] for s in results)
    drp_total = sum(results[s]['drp_pos'] for s in results)
    spk_total = sum(results[s]['spk_pos'] for s in results)
    any_total = sum(results[s]['any_pos'] for s in results)

    print("\n--- Aggregated ---")
    print(f"[{state}] total: N={N_total}, drp+= {drp_total}, spk+= {spk_total}, any+= {any_total}")

    # LaTeX bullet (train/test numbers; includes val inside parens)
    tr, va, te = results['train'], results['val'], results['test']
    text = (
        rf"Train $N={tr['n']}$ (positives: \texttt{{drp}}={tr['drp_pos']}, \texttt{{spk}}={tr['spk_pos']}); "
        rf"Val $N={va['n']}$ (\texttt{{drp}}={va['drp_pos']}, \texttt{{spk}}={va['spk_pos']}); "
        rf"Test $N={te['n']}$ (positives: \texttt{{drp}}={te['drp_pos']}, \texttt{{spk}}={te['spk_pos']})."
    )
    print("\Summary:\n" + text)

# ---- Run for one state (QLD) ----
summarize_split(state='QUN', sequence_length=96)

if __name__ == '__main__':
    ds = ElectricityDemandData()
    #print(ds[15])

    labels_drp = [dict['labels_drp'] for dict in ds]
    labels_spk = [dict['labels_spk'] for dict in ds]

    #print(np.histogram(labels_drp, bins=2)[0])
    #print(np.histogram(labels_spk, bins=2)[0])
