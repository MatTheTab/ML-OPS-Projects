from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning as L
from sklearn.preprocessing import StandardScaler


def find_intersecting_cols(starting_df, all_df):
    all_valid_columns = list(set(starting_df.columns))
    for cylinderi in all_df:
        all_valid_columns = list(
            set(all_valid_columns).intersection(set(cylinderi.columns))
        )
    return all_valid_columns


def find_cols_keyword(cols, keyword):
    keyword = keyword.lower()
    return [col for col in cols if keyword in col.lower()]


def columns_to_triples(all_glo_cols, all_llz_cols, all_lpz_cols):
    col_triples = []
    for col in all_glo_cols:
        num = int(col[3:])
        llz_found = None
        lpz_found = None

        for llz_col in all_llz_cols:
            if int(llz_col[3:]) == num:
                llz_found = llz_col
                break

        for lpz_col in all_lpz_cols:
            if int(lpz_col[3:]) == num:
                lpz_found = lpz_col
                break

        if llz_found is not None and lpz_found is not None:
            col_triples.append([col, llz_found, lpz_found])
    return np.unique(np.array(col_triples), axis=0)


def window_arr(arr, window_size, window_step):
    curr_step = 0
    windowed_arr = []
    while curr_step + window_size < len(arr):
        windowed_arr.append((curr_step, curr_step + window_size))
        curr_step += window_step
    return windowed_arr


class Engine_Dataset(Dataset):
    def __init__(self, glo_data, llz_data, lpz_data, windows):
        self.glo_data = glo_data
        self.llz_data = llz_data
        self.lpz_data = lpz_data
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_start, window_end = self.windows[idx]
        return (
            self.glo_data[window_start:window_end].astype(np.float32),
            self.llz_data[window_start:window_end].astype(np.float32),
            self.lpz_data[window_start:window_end].astype(np.float32),
        )


class EngineDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_arr_glo,
        train_arr_llz,
        train_arr_lpz,
        windows_train,
        val_arr_glo,
        val_arr_llz,
        val_arr_lpz,
        windows_val,
        test_arr_glo,
        test_arr_llz,
        test_arr_lpz,
        windows_test,
        batch_size,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_arr_glo = train_arr_glo
        self.train_arr_llz = train_arr_llz
        self.train_arr_lpz = train_arr_lpz
        self.val_arr_glo = val_arr_glo
        self.val_arr_llz = val_arr_llz
        self.val_arr_lpz = val_arr_lpz
        self.test_arr_glo = test_arr_glo
        self.test_arr_llz = test_arr_llz
        self.test_arr_lpz = test_arr_lpz
        self.windows_train = windows_train
        self.windows_val = windows_val
        self.windows_test = windows_test
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Engine_Dataset(
                self.train_arr_glo,
                self.train_arr_llz,
                self.train_arr_lpz,
                self.windows_train,
            )
            self.val_dataset = Engine_Dataset(
                self.val_arr_glo, self.val_arr_llz, self.val_arr_lpz, self.windows_val
            )

        if stage == "test" or stage is None:
            self.test_dataset = Engine_Dataset(
                self.test_arr_glo,
                self.test_arr_llz,
                self.test_arr_lpz,
                self.windows_test,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15)


def transform_data(train_arr, val_arr, test_arr):
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_arr.reshape(-1, 1)).flatten()
    val_arr = scaler.transform(val_arr.reshape(-1, 1)).flatten()
    test_arr = scaler.transform(test_arr.reshape(-1, 1)).flatten()
    return scaler, train_arr, val_arr, test_arr
