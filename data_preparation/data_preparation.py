import os
from pathlib import Path
from typing import Union, Optional, List, Dict
import pandas as pd


class UCIHARDataset:
    """
    Class to load, parse, internally represent, and export the UCI HAR 
    (Human Activity Recognition) Dataset.

    Parameters
    ----------
    dataset_path : str or Path, default="../../Dataset/UCI_HAR_Dataset"
        Path to the root directory containing the UCI HAR Dataset.
    make_unique_features : bool, default=True
        Whether to append unique numeric suffixes to duplicate feature names in features.txt 
        to ensure clean Pandas DataFrame column indexing.
    """

    def __init__(
        self, 
        dataset_path: Union[str, Path] = "../../Dataset/UCI_HAR_Dataset",
        make_unique_features: bool = True
    ):
        self.dataset_path = Path(dataset_path).resolve()
        self.make_unique_features = make_unique_features

        # Internal DataFrame representations of metadata
        self.features_df: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.activity_labels_df: Optional[pd.DataFrame] = None
        self.activity_map: Dict[int, str] = {}

        # Internal DataFrame representations of data splits
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.full_df: Optional[pd.DataFrame] = None

        # Automatically parse metadata upon initialization
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        Parses features.txt and activity_labels.txt and stores them in internal
        Pandas DataFrames and mappings.
        """
        features_file = self.dataset_path / "features.txt"
        activity_file = self.dataset_path / "activity_labels.txt"

        if not features_file.exists():
            raise FileNotFoundError(f"features.txt not found at: {features_file}")
        if not activity_file.exists():
            raise FileNotFoundError(f"activity_labels.txt not found at: {activity_file}")

        # Load features.txt as Pandas DataFrame
        self.features_df = pd.read_csv(
            features_file, sep=r'\s+', header=None, names=['feature_id', 'feature_name']
        )

        # Process feature names (handling duplicates if requested)
        if self.make_unique_features:
            counts: Dict[str, int] = {}
            unique_names: List[str] = []
            for name in self.features_df['feature_name']:
                if name in counts:
                    counts[name] += 1
                    unique_names.append(f"{name}_{counts[name]}")
                else:
                    counts[name] = 0
                    unique_names.append(name)
            self.feature_names = unique_names
        else:
            self.feature_names = self.features_df['feature_name'].tolist()

        # Load activity_labels.txt as Pandas DataFrame
        self.activity_labels_df = pd.read_csv(
            activity_file, sep=r'\s+', header=None, names=['activity_id', 'activity_name']
        )

        # Store dictionary mapping activity ID (number) -> activity name
        self.activity_map = dict(
            zip(self.activity_labels_df['activity_id'], self.activity_labels_df['activity_name'])
        )

    def build_dataframe(
        self, 
        is_train: Union[bool, str] = True,
        save_to_csv: bool = False,
        output_csv_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Builds a DataFrame for the Training set or Test set according to a flag.

        Parameters
        ----------
        is_train : bool or str, default=True
            Flag indicating split: True or 'train' for training data, False or 'test' for test data.
        save_to_csv : bool, default=False
            If True, automatically exports the generated DataFrame to a CSV file.
        output_csv_path : str or Path, optional
            File path for saving CSV. If None, defaults to '{split}_dataset.csv' in dataset_path.

        Returns
        -------
        pd.DataFrame
            The created Pandas DataFrame containing subject_id, 561 feature columns,
            target_name (second-to-last column), and target (last column).
        """
        if isinstance(is_train, str):
            split = is_train.strip().lower()
            if split not in ['train', 'test']:
                raise ValueError(f"Invalid split string '{is_train}'. Expected 'train' or 'test'.")
        elif isinstance(is_train, bool):
            split = "train" if is_train else "test"
        else:
            raise TypeError(f"is_train flag must be bool or str ('train'/'test'), got {type(is_train)}")

        split_dir = self.dataset_path / split
        x_file = split_dir / f"X_{split}.txt"
        y_file = split_dir / f"y_{split}.txt"
        sub_file = split_dir / f"subject_{split}.txt"

        for file_path in [x_file, y_file, sub_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Internally represent text files as DataFrames
        X_df = pd.read_csv(x_file, sep=r'\s+', header=None, names=self.feature_names)
        y_df = pd.read_csv(y_file, header=None, names=['target'])
        y_df['target_name'] = y_df['target'].map(self.activity_map)
        sub_df = pd.read_csv(sub_file, header=None, names=['subject_id'])

        # Concatenate into single unified DataFrame
        # subject_id first, 561 features next, target_name as last - 1, target as last column
        df = pd.concat([sub_df, X_df, y_df[['target_name', 'target']]], axis=1)

        # Store internally
        if split == "train":
            self.train_df = df
        else:
            self.test_df = df

        if save_to_csv:
            if output_csv_path is None:
                output_csv_path = self.dataset_path / f"{split}_dataset.csv"
            self.save_csv(df, output_path=output_csv_path)

        return df

    def save_csv(
        self, 
        df: Optional[pd.DataFrame] = None, 
        is_train: Optional[Union[bool, str]] = None, 
        output_path: Optional[Union[str, Path]] = None,
        index: bool = False
    ) -> Path:
        """
        Stores the dataset in a CSV file.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The DataFrame to store. If None, uses train_df or test_df depending on is_train.
        is_train : bool or str, optional
            Flag to select split to build/save if df is not provided.
        output_path : str or Path, optional
            Destination CSV path. Defaults to dataset_path / 'uci_har_dataset.csv'.
        index : bool, default=False
            Whether to include row indices in the exported CSV.

        Returns
        -------
        Path
            Path of the saved CSV file.
        """
        if df is None:
            if is_train is not None:
                df = self.build_dataframe(is_train=is_train)
            elif self.train_df is not None:
                df = self.train_df
            elif self.test_df is not None:
                df = self.test_df
            else:
                df = self.build_dataframe(is_train=True)

        if output_path is None:
            output_path = self.dataset_path / "uci_har_dataset.csv"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=index)
        return output_path

    def build_full_dataframe(
        self, 
        save_to_csv: bool = False, 
        output_csv_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Combines both training (7352 samples) and testing (2947 samples) sets
        into a single complete DataFrame of 10299 samples and 564 columns.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all 10299 samples.
        """
        train = self.build_dataframe(is_train=True)
        test = self.build_dataframe(is_train=False)
        self.full_df = pd.concat([train, test], axis=0, ignore_index=True)

        if save_to_csv:
            if output_csv_path is None:
                output_csv_path = self.dataset_path / "full_dataset.csv"
            self.save_csv(self.full_df, output_path=output_csv_path)

        return self.full_df


# Alias for alternative naming preference
UCIHARDataLoader = UCIHARDataset
DATASET_PATH = "/Users/calogeroforte/work/Python_work_area/9_1_Test_ML_DL/Dataset/UCI_HAR_Dataset"

if __name__ == "__main__":
    uci = UCIHARDataLoader(dataset_path=DATASET_PATH)
    train_df = uci.build_dataframe(is_train=True, save_to_csv=False)
    # test_df = uci.build_dataframe(is_train=False, save_to_csv=False)
    # full_df = uci.build_full_dataframe(save_to_csv=False)
    # print(train_df.head())
    # print(test_df.head())
    # print(full_df.head())
    uci.save_csv(train_df, is_train=True, output_path="/Users/calogeroforte/work/Python_work_area/9_1_Test_ML_DL/Code/Exam/Dataset/train_dataset.csv")
    
