import argparse
from uuid import uuid4

import pandas as pd

from ami.mp.configuration import Configuration
from ami.data_manager import InMemoryDataManager
from ami.scheduler import SerialSchedulerFactory
from ami.worker import ShareMemorySingleThreadWorkerFactory
from ami.worker_pool import SingleNodeWorkerPoolFactory
from ami.option import Some

from surrogate.acquisition import EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import ExpectedImprovementRanker, RandomRanker
from raspa import Adsorption

# --------------------------------------------------------------------------------------- 

# Define your starting files

FEATURES_CSV = 'FEATURES.csv'
PRIOR_CSV = 'PRIOR.csv' # If you don't have prior data, 'python main.py -p 0'

# ---------------------------------------------------------------------------------------

# Collect args

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='Total number of structures to screen.', default=20)
parser.add_argument('-r', type=str, help='Ranker to use', default='gp')
parser.add_argument('-p', type=int, help='Load prior dataset - 1 (True) or 0 (False)', default=1)
args = parser.parse_args()

code = uuid4().hex[::4]
pool_size = 10

n_tasks = args.n
ranker_choice = args.r
prior_true = args.p

run_code = F'{ranker_choice}_{code}'

# ---------------------------------------------------------------------------------------

# Converts my csv to hdf5 AND standardises features (Gaussian with 0 mean and unit variance)

import h5py
from sklearn.preprocessing import StandardScaler

# Load your CSV
df = pd.read_csv(FEATURES_CSV)

# Remove the cif_path column from features
features_df = df.drop(columns=['cif_path'])

# Extract features as numpy array
features = features_df.values

# Standardise features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Save to HDF5
with h5py.File(f'{FEATURES_CSV}.hdf5', 'w') as f:
    f.create_dataset('X', data=scaled_features)

# --------------------------------------------------------------------------------------- 

# Set up ML code
hdf5_dataset = Hdf5Dataset(f'{FEATURES_CSV}.hdf5')
 
model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

gp_ranker = ExpectedImprovementRanker(
    model=DenseGaussianProcessregressor(data_set=hdf5_dataset),
    acquisitor=EiRanking()
    )

rf_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(data_set=hdf5_dataset),
    acquisitor=EiRanking()
)

init_ranker = RandomRanker()

surrogate_ranker = {'gp': gp_ranker, 'rf': rf_ranker, 'rr': init_ranker}[ranker_choice]


# ---------------------------------------------------------------------------------------

# Set up AMI code

calc = Adsorption.from_template_folder(F"internal_workdir_{run_code}", "raspa_template")
pool = SingleNodeWorkerPoolFactory()
pool.set("ncpus", pool_size)

# Load data manager first to access its schema

data_manager = InMemoryDataManager.from_indexed_list_in_file(FEATURES_CSV,
                                                       calc_schema=calc.schema(),
                                                       surrogate_schema=surrogate_ranker.schema(),
                                                       csv_filename=F'ami_output_{run_code}.csv'
                                                       )

config = Configuration(
    scheduler=SerialSchedulerFactory(),
    worker=ShareMemorySingleThreadWorkerFactory(),
    data=data_manager,
    truth=calc,
    pool=pool,
    initial_ranker=init_ranker,
    ranker=surrogate_ranker,
)

# ---------------------------------------------------------------------------------------

# Set up prior dataset

if prior_true == 1:
    prior_values = pd.read_csv(PRIOR_CSV)
    X_init, y_init = prior_values['index'].tolist(), prior_values['value'].tolist()

    for x_, y_ in zip(X_init, y_init):
        try:
            # Manually select the item first
            config.data.state.select(x_)

            # Then set the result
            result = config.data.set_prior_result(x_, Some(y_))
            #print(f"Set result for index {x_}")

        except Exception as e:
            print(f"Exception for index {x_}: {e}")
    
    print(f"Initialised AMI  using {len(X_init)} prior data points.")
    
else:
    print(f"Starting AMI without prior data.")

# ---------------------------------------------------------------------------------------

# Run screening

runner = config.build()
runner.run(50)

# ---------------------------------------------------------------------------------------
