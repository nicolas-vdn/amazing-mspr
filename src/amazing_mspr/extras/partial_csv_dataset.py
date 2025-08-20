import numpy as np
import pandas as pd
from kedro.io import AbstractDataset


class PartialCSVDataSet(AbstractDataset):
    def __init__(self, filepath: str, usecols=None, nrows=None, random=False, **kwargs):
        self._filepath = filepath
        self._usecols = usecols
        self._nrows = nrows
        self._random = random
        self._kwargs = kwargs

    def _load(self) -> pd.DataFrame:
        if self._random and self._nrows is not None:
            # Compter le nombre total de lignes (hors header)
            total_lines = sum(1 for _ in open(self._filepath)) - 1

            # Tirer aléatoirement nrows indices
            skip = sorted(
                np.random.choice(
                    np.arange(1, total_lines + 1),  # lignes après le header
                    size=total_lines - self._nrows,  # celles qu’on *ignore*
                    replace=False
                )
            )
        else:
            skip = None

        return pd.read_csv(
            self._filepath,
            usecols=self._usecols,
            nrows=None if self._random else self._nrows,  # random → pas de nrows direct
            skiprows=skip,
            **self._kwargs
        )

    def _save(self, data) -> None:
        raise NotImplementedError("Ce dataset est en lecture seule.")

    def _describe(self):
        return dict(filepath=self._filepath,
                    usecols=self._usecols,
                    nrows=self._nrows,
                    random=self._random)
