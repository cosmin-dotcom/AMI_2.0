from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import h5py


# -----------------------------------------------------------------------------------------------------------------------------


class Hdf5Dataset:
    """hdf5 dataset containing features for MOFs.
    Loads features from HDF5 dataset for passed indices.
    Useful for scenarios where indices need to be passed in place of explicit features.
    """
    
    def __init__(self, hdf5_loc: str, data_key: str='X') -> None:
        self.hdf5_loc = str(hdf5_loc)
        self.data_key = str(data_key)
        
    def __repr__(self) -> str:
        return F'Hdf5Dataset(hdf5_loc="{self.hdf5_loc}", data_key="{self.data_key}")'
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self[:].shape
    
    def __len__(self) -> int:
        return self.shape[0]
        
    def __getitem__(self, k: NDArray[np.int_]) -> NDArray[NDArray]:
        """return features from hdf5 dataset for passed indices.

        Parameters
        ----------
        k : NDArray[np.int_]
            indices to load data from 

        Returns
        -------
        NDArray[NDArray]
            feature matrix where each row are the features of the specified index.
        """
        k = np.asarray(k).ravel()
        
        with h5py.File(self.hdf5_loc, 'r') as f:
            X_ = f[self.data_key]
            
            if isinstance(k[0], slice):  # first index since convert to array above
                X = X_[:]                # return entire dataset
            else:
                X = np.vstack([X_[idx] for idx in k])  # avoids worries of out of order indexing / duplicate indices being passed
            
        if X.ndim == 1:
            X = X.reshape(1, -1)         # ensure always 2d output for simplicity
            
        return X    

# -----------------------------------------------------------------------------------------------------------------------------
