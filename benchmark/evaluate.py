"""Evaluates the model.

Usage: 
    python evaluate.py
Use --help command line option for more information.
"""
import argparse
from datetime import datetime, timedelta
import pathlib
import random
import warnings

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_geometric


def seed_everything(seed: int):
    """Seeds all relevant random number generators.
    
    Args:
        seed (int): Number used for seeding.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    random.seed(seed)


class Args:
    """Container class for the command line arguments.
    """
    seed: int
    batch_size: int
    dataset_dir: str
    use_seed: bool
    model_path: str


def parse_args() -> Args:
    """Parses command line arguments and returns them in a namespace.

    Returns:
        Args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help=
        'The number used to seed pseudo-random number generators. Defaults to 42.',
        default=42)
    parser.add_argument('--no-seed',
                        dest='use_seed',
                        action='store_false',
                        help='Do not seed any generators. Overrides --seed.')
    parser.add_argument(
        '--dataset-dir',
        dest='dataset_dir',
        help=
        'The directory with the MovieLens 100K dataset. Defaults to the ./data/raw/mk-100k/.',
        type=str,
        default='./data/raw/ml-100k/')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='Batch size used during training. Defaults to 64.',
        type=int,
        default=64)
    parser.add_argument(
        '--model-path',
        dest='model_path',
        help=
        'Path to the model contatining the model state dictionary. Defaults to ./models/best.pt',
        type=str,
        default='./models/best.pt')
    args = parser.parse_args()
    return args


class RatingDataset(Dataset):
    """Dataset containing tuples of `user_id, item_id, rating`.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """Dataset containing tuples of `user_id, item_id, rating`.

        Args:
            dataframe (DataFrame): The dataset containing columns
            `user_id`, `item_id`, and `rating`.
        """
        super().__init__()
        self.items = list(
            zip(dataframe['user_id'] - 1, dataframe['item_id'] - 1,
                dataframe['rating']))

    def __len__(self) -> int:
        """Calculates the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        """Returns the `index`-th item in the dataset.

        Args:
            index (int): The index of the item requested.

        Returns:
            tuple[int, int, int]: Tuple of `user_id`, `item_id`, `rating`.
            `rating` ranges from 1 to 5 inclusive.
        """
        return self.items[index]


class UserEncoding:
    """Encodes user information into numerical format. Age is rescaled to be between
    0 and 1, gender and occupation are one-hot encoded.
    """

    def __init__(self):
        """Encodes user information into numerical format. Age is rescaled to be between
        0 and 1, gender and occupation are one-hot encoded.
        """
        self.age_scaler = sklearn.preprocessing.MinMaxScaler()
        self.gender_encoder = sklearn.preprocessing.LabelEncoder()
        self.occupation_encoder = sklearn.preprocessing.LabelEncoder()
        self.occupation_one_hot = sklearn.preprocessing.OneHotEncoder(
            sparse=False)

    def fit(self, dataframe: pd.DataFrame):
        """Fits all the preprocessing sklearn modules on the data.

        Args:
            dataframe (DataFrame): Dataframe containing the user information.
        """
        self.age_scaler.fit(dataframe.age.to_numpy().reshape(-1, 1))
        self.gender_encoder.fit(dataframe.gender.to_numpy().reshape(-1, 1))
        occupation = self.occupation_encoder.fit_transform(
            dataframe.occupation.to_numpy().reshape(-1, 1)).reshape(-1, 1)
        self.occupation_one_hot.fit(occupation)

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Transforms the user information and returs a numpy array
        with encoded values.

        Args:
            dataframe (DataFrame): Dataframe with user information.

        Returns:
            ndarray: Encoded user information.
        """
        age = self.age_scaler.transform(dataframe.age.to_numpy().reshape(
            -1, 1))
        gender = self.gender_encoder.transform(
            dataframe.gender.to_numpy().reshape(-1, 1)).reshape(-1, 1)
        occupation = self.occupation_encoder.transform(
            dataframe.occupation.to_numpy().reshape(-1, 1)).reshape(-1, 1)
        occupation = self.occupation_one_hot.transform(occupation)
        user_features = np.concatenate((age, gender, occupation), axis=1)
        return user_features

    def fit_transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Fits all preprocessing modules and transforms the user information,
        then returs a numpy array with encoded values.

        Args:
            dataframe (DataFrame): Dataframe with user information.

        Returns:
            ndarray: Encoded user information.
        """
        age = self.age_scaler.fit_transform(dataframe.age.to_numpy().reshape(
            -1, 1))
        gender = self.gender_encoder.fit_transform(
            dataframe.gender.to_numpy().reshape(-1, 1)).reshape(-1, 1)
        occupation = self.occupation_encoder.fit_transform(
            dataframe.occupation.to_numpy().reshape(-1, 1)).reshape(-1, 1)
        occupation = self.occupation_one_hot.fit_transform(occupation)
        user_features = np.concatenate([age, gender, occupation], axis=1)
        return user_features


class ItemEncoding:
    """Encodes item information into numerical format. Date information is encoded
    and scaled to be between 0 and 1.
    """

    def __init__(self):
        """Encodes item information into numerical format. Date information is encoded
        and scaled to be between 0 and 1.
        """
        self.time_scaler = sklearn.preprocessing.MinMaxScaler()

    def _parse_date(self, date_string: str) -> float:
        """Parses a date in format "DD-MMM-YYYY", e.g. "01-Jan-1995", and returns the time
        difference between the given date and Jan 1st 1900 in seconds (the date does not
        matter, as the timestamp is scaled afterwards.)

        Args:
            date_string (str): String containing the date in specified format.

        Returns:
            float: Number of seconds passed from January 1st 1900 00:00:00 to the given date.
        """
        parsed_date = datetime.strptime(date_string, '%d-%b-%Y')
        start = datetime(1900, 1, 1)
        return (parsed_date - start) / timedelta(seconds=1)

    def fit(self, dataframe: pd.DataFrame):
        """Fits all preprocessing modules on the given item information.

        Args:
            dataframe (DataFrame): Dataframe with item information.

        Returns:
            ndarray: Encoded item information.
        """
        timestamps = dataframe['release date'].fillna('01-Jan-1995').apply(
            self._parse_date).to_numpy().reshape(-1, 1)
        self.time_scaler.fit(timestamps)

    def transform(self, dataframe: pd.DataFrame,
                  genre_columns: list[str]) -> np.ndarray:
        """Transforms the item information, then returs a numpy array
        with encoded values.

        Args:
            dataframe (DataFrame): Dataframe with item information.
            genres (list[str]): Names of columns containing genre information.

        Returns:
            ndarray: Encoded item information.
        """
        timestamps = dataframe['release date'].fillna('01-Jan-1995').apply(
            self._parse_date).to_numpy().reshape(-1, 1)
        scaled_timestamps = self.time_scaler.transform(timestamps)
        genres = dataframe[genre_columns].to_numpy()
        item_features = np.concatenate([scaled_timestamps, genres], axis=1)
        return item_features

    def fit_transform(self, dataframe: pd.DataFrame,
                      genre_columns: list[str]) -> np.ndarray:
        """Fits all preprocessing modules and transforms the item information,
        then returs a numpy array with encoded values.

        Args:
            dataframe (DataFrame): Dataframe with item information.
            genres (list[str]): Names of columns containing genre information.

        Returns:
            ndarray: Encoded item information.
        """
        timestamps = dataframe['release date'].fillna('01-Jan-1995').apply(
            self._parse_date).to_numpy().reshape(-1, 1)
        scaled_timestamps = self.time_scaler.fit_transform(timestamps)
        genres = dataframe[genre_columns].to_numpy()
        item_features = np.concatenate([scaled_timestamps, genres], axis=1)
        return item_features


class RatingEstimator(nn.Module):
    """The model for estimating a rating a user is most likely to give a movie.
    """

    def __init__(self, user_encodings: torch.Tensor,
                 item_encodings: torch.Tensor):
        """The model for estimating a rating a user is most likely to give a movie.

        Args:
            user_encodings (Tensor): Matrix, containing encoded user information.
            Shaped `(n_users, user_dim)`
            item_encodings (Tensor): Matrix, containing encoded item information.
            Shaped `(n_items, item_dim)`
        """
        super().__init__()
        n_users = user_encodings.shape[0]
        n_items = item_encodings.shape[0]
        self.register_buffer('item_encodings', item_encodings)
        self.register_buffer('user_encodings', user_encodings)
        self.user_embed = nn.Embedding(n_users, 20)
        self.item_embed = nn.Embedding(n_items, 20)
        self.user_fc = nn.Linear(20 + user_encodings.shape[1], 50)
        self.item_fc = nn.Linear(20 + item_encodings.shape[1], 50)

    def forward(self, user_ids: torch.Tensor,
                item_ids: torch.Tensor) -> torch.Tensor:
        """Predicts ratings users with given IDs are most likely to give
        to items with given IDs.

        Args:
            user_ids (torch.Tensor): Tensor with user IDs, shaped `(n,)`.
            item_ids (torch.Tensor): Tensor with item IDs, shaped `(n,)`.

        Returns:
            Tensor: Predicted ratings of the movies. Shaped `(n,)`
        """
        user_encodings = self.user_encodings[user_ids]
        item_encodings = self.item_encodings[item_ids]
        user_embeddings = self.user_embed(user_ids)
        item_embeddings = self.item_embed(item_ids)
        users = self.user_fc(
            torch.concat([user_encodings, user_embeddings], dim=-1))
        items = self.item_fc(
            torch.concat([item_encodings, item_embeddings], dim=-1))
        ratings = (users * items).sum(dim=-1)
        return users, items, ratings

    def recommend(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Returns the predicted ratings for all items in the dataset.

        Args:
            user_ids (torch.Tensor): The users to predict ratings for, shaped `(n,)`.

        Returns:
            torch.Tensor: Predicted ratings for all items in the dataset, shaped `(n, n_items)`.
        """
        item_ids = torch.arange(len(self.item_encodings))
        user_encodings = self.user_encodings[user_ids]
        item_encodings = self.item_encodings[item_ids]
        user_embeddings = self.user_embed(user_ids)
        item_embeddings = self.item_embed(item_ids)
        users = self.user_fc(
            torch.concat([user_encodings, user_embeddings], dim=-1))
        items = self.item_fc(
            torch.concat([item_encodings, item_embeddings], dim=-1))
        ratings = torch.matmul(users, torch.transpose(items, 0, 1))
        return ratings


def evaluate_model(loader: DataLoader, model: RatingEstimator) -> float:
    """Evaluates the model on the given dataset and returns the RMSE metric.

    Args:
        loader (DataLoader): The dataloader of the evaluation set.
        model (RatingEstimator): Model to be evaluated.

    Returns:
        float: RMSE metric of the model on the dataset.
    """
    total_len = 0
    total_mse = 0
    model.eval()
    with torch.no_grad():
        for user_ids, item_ids, ratings in loader:
            _users, _items, pred_ratings = model(user_ids, item_ids)
            mse = torch.square(ratings - pred_ratings).mean()
            batch_len = len(ratings)
            total_len += batch_len
            total_mse += batch_len * mse
    mse = total_mse / total_len
    return torch.sqrt(mse).item()


def main():
    """Entry point of the script.
    """
    args = parse_args()
    dataset_dir = pathlib.Path(args.dataset_dir)

    ratings_df = pd.read_csv(
        dataset_dir / 'u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'])
    dataset = RatingDataset(ratings_df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # The movie data is loaded
    item_columns = [
        'movie id', 'movie title', 'release date', 'video release date',
        'IMDb URL'
    ]
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    items_df = pd.read_csv(dataset_dir / 'u.item',
                           sep='|',
                           names=item_columns + genre_columns,
                           encoding="windows-1252")

    # The user data is loaded.
    user_columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    users_df = pd.read_csv(dataset_dir / 'u.user',
                           sep='|',
                           names=user_columns,
                           encoding="windows-1252")

    encode_users = UserEncoding()
    user_encodings = encode_users.fit_transform(users_df)
    user_encodings = torch.tensor(user_encodings).float()

    encode_items = ItemEncoding()
    item_encodings = encode_items.fit_transform(items_df, genre_columns)
    item_encodings = torch.tensor(item_encodings).float()

    model_path = pathlib.Path(args.model_path)
    model = RatingEstimator(user_encodings, item_encodings)
    model.load_state_dict(torch.load(model_path))

    rmse = evaluate_model(loader, model)

    print(f'The model {model_path} scored {rmse:.3f} RMSE.')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
