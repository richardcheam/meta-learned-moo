import torch

# Image dataset adapter
class LibMoonDatasetAdapter(torch.utils.data.Dataset):
    """
    Adapter that makes LibMOON datasets compatible with
    the dSprites-style Dataset used by the hypernetwork training code.
    """

    def __init__(self, libmoon_dataset):
        self.ds = libmoon_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # image: [1, 36, 36]
        img = sample['data']

        # labels: [n_tasks] -> [2]
        ys = torch.stack(
            [sample['labels_l'], sample['labels_r']],
            dim=0
        )

        return img, ys

# Tabular dataset adapter
class LibMoonTabularAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = item["data"] #shape (D,)
        y = item["labels"] #scalar
        s = item["sensible_attribute"] #scalar

        # task 0 = prediction
        # task 1 = fairness (uses same y + s internally)
        ys = torch.stack([y, s])  # shape (2,)

        return x, ys

# Temporal dataset adapter (Electricity Demand, ...)
class LibMoonTemporalAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        dataset: ElectricityDemandData from LibMOON
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # time series input: shape (T,)
        x = item["data"]              # tensor [sequence_length]

        # two anomaly labels
        y_drp = item["labels_drp"]    # task 0
        y_spk = item["labels_spk"]    # task 1

        ys = torch.stack([y_drp, y_spk])  # shape (2,)

        return x, ys