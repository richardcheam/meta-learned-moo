import torch
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset:
    def __init__(self, path, tasks_ids: int, val_size=0):
        self.path = path
        self.test_size = 0.1
        self.tasks_ids = tasks_ids
        self.val_size = val_size

    def get_datasets(self):        
        trainData = np.load(self.path)
        trainX = trainData['imgs']
        trainLabel = trainData['latents_classes'][:, self.tasks_ids]
        n_train = len(trainX)

        # for type checking
        testX = []
        testLabel = []
        n_test = 0 
        valX = []
        valLabel = []
        n_val = 0

        if self.test_size > 0:
            trainX, testX, trainLabel, testLabel = train_test_split(
                    trainX, trainLabel, test_size=self.test_size, random_state=42
                    )
            n_train = len(trainX)
            n_test = len(testX)

        if self.val_size > 0:
            trainX, valX, trainLabel, valLabel = train_test_split(
                trainX, trainLabel, test_size=self.val_size, random_state=42
            )
            n_train = len(trainX)
            n_val = len(valX)

        # Images 
        trainX = torch.from_numpy(trainX.reshape(n_train, 1, 64, 64)).float()
        trainLabel = torch.from_numpy(trainLabel).long()
        testX = torch.from_numpy(testX.reshape(n_test, 1, 64, 64)).float()
        testLabel = torch.from_numpy(testLabel).long()

        train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
        test_set = torch.utils.data.TensorDataset(testX, testLabel)

        if self.val_size > 0:
            valX = torch.from_numpy(valX.reshape(n_val, 1, 64, 64)).float()
            valLabel = torch.from_numpy(valLabel).long()
            val_set = torch.utils.data.TensorDataset(valX, valLabel)
            print(f"train_size: {n_train}")
            print(f"val_size: {n_val}")
            print(f"test_size: {n_test}")
            return train_set, val_set, test_set
        print(f"train_size: {n_train}")
        print(f"test_size: {n_test}")
        return train_set, test_set
