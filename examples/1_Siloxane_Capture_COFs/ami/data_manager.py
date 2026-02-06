from dataclasses import dataclass, MISSING
from io import IOBase
from pathlib import Path
from typing import Collection, Sequence, Tuple, List, Union
import csv

import numpy as np

import ami.abc
from ami.abc import Target, Feature
from ami.abc.calculator import OpaqueParameters
from ami.option import Option, Nothing, Some
from ami.result import Result, Ok
from ami.schema import Schema

Index = int


@dataclass(slots=True, frozen=True)
class InMemoryStateMachine(ami.abc.StateMachineInterface):
    available: np.ndarray[bool] = MISSING
    done: np.ndarray[bool] = MISSING
    failed: np.ndarray[bool] = MISSING

    @classmethod
    def from_size(cls, size: int):
        return cls(
            available=np.ones(size, dtype=bool),
            done=np.zeros(size, dtype=bool),
            failed=np.zeros(size, dtype=bool)
        )

    def __post_init__(self):
        assert len(self) == len(self.failed)
        assert len(self) == len(self.done)
        assert len(self) == len(self.available)
        assert np.sum((~self.done) & self.failed) == 0

    def _is_selectable(self, index: Index) -> bool:
        print("SINGLE INDEX:", index)
        return (not self.done[index]) and (self.available[index]) and (not self.failed[index])

    def _is_settable(self, index: Index) -> bool:
        return (not self.done[index]) and (not self.available[index]) and (not self.failed[index])

    def select(self, index: Index) -> None:
        if not self._is_selectable(index):
            raise RuntimeError(f"Tried to select unselectable item at index '{index}'.")
        self.available[index] = False

    def set(self, index: Index, success: bool) -> None:
        if not self._is_settable(index):
            raise RuntimeError(f"Tried to set unsettable item at index '{index}'.")
        self.done[index] = True
        self.failed[index] = not success

    def reset(self, index: Index) -> None:
        self.done[index] = False
        self.failed[index] = False
        self.available[index] = True

    def list_done(self, include_failures=False) -> Collection[bool]:
        if include_failures:
            # All done, failed or not
            return self.done
        else:
            # All not failed
            return (~self.failed) & self.done

    def list_available(self) -> Collection[bool]:
        return (~self.done) & self.available

    def __len__(self) -> int:
        return len(self.done)


@dataclass(slots=True, frozen=True)
class IndexedSingleFloatTargetSurrogateProvider(ami.abc.SurrogateProviderInterface):
    features: np.ndarray = MISSING
    targets: np.ndarray = MISSING
    _schema: ami.abc.SchemaInterface = MISSING

    @classmethod
    def from_size_and_schema(cls, size: int):
        features = np.arange(size, dtype=int)
        targets = np.empty(size, dtype=float)
        schema = Schema(input_schema=[('index', int)], output_schema=[('target', float)])
        return cls(
            features=features,
            targets=targets,
            _schema=schema
        )

    def __post_init__(self):
        assert len(self.features) == len(self.targets)
        assert len(self.features) == len(self)

    def known(self, state: ami.abc.StateMachineInterface) -> Tuple[Sequence[Feature], Sequence[Target]]:
        done: Collection[bool] = state.list_done(include_failures=False)
        return self.features[done], self.targets[done]

    def unknown(self, state: ami.abc.StateMachineInterface) -> Sequence[Feature]:
        available: Collection[bool] = state.list_available()
        return self.features[available]

    def set_target(self, index: Index, value: Option[Target]) -> None:
        match value:
            case Some(v):
                self.targets[index] = v
            case Nothing:
                pass

    def __len__(self):
        return len(self.features)

    def schema(self) -> ami.abc.SchemaInterface:
        return self._schema


@dataclass(slots=True, frozen=True)
class FileStreamerTruthProvider(ami.abc.TruthProviderInterface):
    filenames: List[Path]
    _schema: ami.abc.SchemaInterface

    def parameters(self, index: Index, state: ami.abc.StateMachineInterface) -> Option[OpaqueParameters]:
        if index >= len(self):
            return Nothing
        state.select(index)
        fpath = self.filenames[index]
        data = fpath.read_bytes()
        return Some({"cif_content": data, "subdir": str(index)})

    def __len__(self) -> int:
        return len(self.filenames)

    def schema(self) -> ami.abc.SchemaInterface:
        return self._schema

    @classmethod
    def from_list_in_file(cls, path: Union[str, Path], schema: ami.abc.SchemaInterface):
        filenames = []
        with Path(path).open(mode="r") as fd:
            for line in fd:
                p = Path(line.strip())
                #FIXME: do proper logging here
                assert p.exists()
                filenames.append(p)
        return cls(filenames=filenames, _schema=schema)

    @classmethod
    def from_csv_file(cls, path: Union[str, Path], schema: ami.abc.SchemaInterface):
        """Create from CSV file where first column contains file paths"""
        filenames = []
        csv_path = Path(path)
        with csv_path.open(mode="r") as fd:
            reader = csv.reader(fd)
            next(reader)  # Skip header row
            for row in reader:
                if row and len(row) > 0:  # Skip empty rows
                    file_path_str = row[0].strip()
                    p = Path(file_path_str)
                    #FIXME: do proper logging here
                    assert p.exists(), f"File does not exist: {p}"
                    filenames.append(p)
        return cls(filenames=filenames, _schema=schema)


@dataclass(slots=True, frozen=True)
class CsvPersistence:
    writer: IOBase
    feature_headers: List[str]
    feature_data: List[List[str]]

    @classmethod
    def from_filename(cls, path: Union[str, Path], features_csv_path: Union[str, Path] = None):
        path = Path(path)
        writer = path.open(mode="w")
        
        feature_headers = []
        feature_data = []
        
        if features_csv_path:
            # Load feature headers and data from CSV
            features_path = Path(features_csv_path)
            with features_path.open(mode="r") as fd:
                reader = csv.reader(fd)
                headers = next(reader)
                feature_headers = headers[1:]  # Skip first column (file paths)
                
                for row in reader:
                    if row and len(row) > 1:  # Skip empty rows and ensure we have data beyond first column
                        feature_data.append(row[1:])  # Skip first column (file paths)
        
        return cls(writer=writer, 
        feature_headers=feature_headers, 
        feature_data=feature_data 
        )

    def __post_init__(self):
        # Write headers including features
        if self.feature_headers:
            # Write the actual column headers
            column_headers = ["index","target","D4 ads load [cm3/cm3]","D4 ads enthalpy [kJ/mol]","Tip5p Henry constant [mol/kg/Pa]"] + self.feature_headers
            print(",".join(column_headers), file=self.writer)
        else:
            print("#Failed to read in columns from features.csv", file=self.writer)

    def __del__(self):
        if not self.writer.closed:
            self.writer.flush()
            self.writer.close()

    def append_valid_result(self, index: Index, value):            #value is a dict so we need to iterate to add output parts
        # Build output line with calculations and features
        dictionary = value
        calc_data = []
        
        for key in dictionary:
            calc_data.append(str(dictionary[key]))
        output_parts = [str(index)] + calc_data
        
        if self.feature_data and index < len(self.feature_data):
            output_parts.extend(self.feature_data[index])
        elif self.feature_headers:
            # Fill with empty values if feature data is missing for this index
            output_parts.extend([''] * len(self.feature_headers))
        
        output_line = ",".join(output_parts)
        print(output_line, file=self.writer)
        self.writer.flush()

    def append_invalid_result(self, index: Index):
        # Build output line with features for invalid results
        output_parts = [f"#{index}", ""]
        
        if self.feature_data and index < len(self.feature_data):
            output_parts.extend(self.feature_data[index])
        elif self.feature_headers:
            # Fill with empty values if feature data is missing for this index
            output_parts.extend([''] * len(self.feature_headers))
        
        output_line = ",".join(output_parts)
        print(output_line, file=self.writer)


@dataclass(slots=True, frozen=True)
class InMemoryDataManager(ami.abc.DataManagerInterface):
    state: ami.abc.StateMachineInterface = MISSING
    surrogate: ami.abc.SurrogateProviderInterface = MISSING
    truth: ami.abc.TruthProviderInterface = MISSING
    io: CsvPersistence = MISSING

    @classmethod
    def from_indexed_list_in_file(cls, 
                                  path: Union[str, Path], 
                                  calc_schema: ami.abc.SchemaInterface,
                                  surrogate_schema: ami.abc.SchemaInterface,
                                  csv_filename: Union[str, Path]='AMI.out',
                                  features_csv_path: Union[str, Path] = None
                                  ):
        """
        Create from either a CSV file with features+paths or a text file with paths.
        
        Args:
            path: Path to input file (either .txt with paths or .csv with features)
            calc_schema: Schema for calculations
            surrogate_schema: Schema for surrogate
            csv_filename: Output CSV filename
            features_csv_path: Optional path to features CSV (if different from path)
        """
        path = Path(path)
        assert path.exists()
        
        # Determine if we're using CSV features
        if features_csv_path or path.suffix.lower() == '.csv':
            csv_path = features_csv_path if features_csv_path else path
            truth = FileStreamerTruthProvider.from_csv_file(path, calc_schema)
            io = CsvPersistence.from_filename(csv_filename, csv_path)
        else:
            # Original behavior for .txt files
            truth = FileStreamerTruthProvider.from_list_in_file(path, calc_schema)
            io = CsvPersistence.from_filename(csv_filename)
        
        size = len(truth)
        surrogate = IndexedSingleFloatTargetSurrogateProvider.from_size_and_schema(size)
        state = InMemoryStateMachine.from_size(size)
        
        return cls(
            state=state,
            surrogate=surrogate,
            truth=truth,
            io=io
        )

    def available_for_calculation(self) -> Sequence[Index]:
        return np.flatnonzero(self.state.list_available())

    def set_result(self, index: Index, value: Option[Target]) -> Result[..., Exception]:
        match value:
            case Some(v):                      #v is a dictionary from Adsorption.calculator
                self.state.set(index, True)
                self.surrogate.set_target(index, Some(v['target']))
                self.io.append_valid_result(index, v)
            case Nothing:
                self.state.set(index, False)
                self.io.append_invalid_result(index)
                print("DATA_MANAGER.PY says: The dict is not being written in raspa.py")
        return Ok(())

    def unknown(self) -> Sequence[Feature]:
        return self.surrogate.unknown(self.state)

    def known(self) -> Tuple[Sequence[Feature], Sequence[Target]]:
        return self.surrogate.known(self.state)

    def __len__(self) -> int:
        return len(self.state)

    def parameters(self, index: Index) -> Option[OpaqueParameters]:
        return self.truth.parameters(index, self.state)