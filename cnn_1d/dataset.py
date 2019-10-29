import xlrd
import numpy as np
import torch


# class Dataset(Dataset):
#     def __init__(self, path):
#         self.trans = transforms.ToTensor()
#         workbook = xlrd.open_workbook(path, encoding_override="utf-8")
#         sheets = workbook.sheet_names()
#         booksheets = workbook.sheet_by_name(sheets[0])
#         col_value = np.array(booksheets.col_values(8), dtype=np.float64)
#         max_value = np.max(col_value)
#         self.data = col_value/max_value
#         print(self.data)
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, item):
#         if item>self.data.shape[0]-11

class Dataset:
    def __init__(self, path):
        workbook = xlrd.open_workbook(path, encoding_override="utf-8")
        sheets = workbook.sheet_names()
        booksheets = workbook.sheet_by_name(sheets[0])
        col_value = np.array(booksheets.col_values(8), dtype=np.float64)
        max_value = np.max(col_value)
        self.data = col_value / max_value

    def getData(self):
        return self.data


if __name__ == '__main__':
    path = "../cnn_1d/data/data2.xlsx"
    dataset = Dataset(path)
    data = dataset.getData()
    print(data.shape[0])
