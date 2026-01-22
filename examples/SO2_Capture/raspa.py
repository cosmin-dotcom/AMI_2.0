from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Tuple, Type, Union

import numpy as np
from ase.io import read

import ami.abc
from ami.abc import SchemaInterface
from ami.result import Result, Ok
from ami.schema import Schema

from ami.serialized_opaque import SerializedOpaque
from ami.abc import OpaqueParameters, OpaqueResults
from multiprocessing import Lock

def find_minimum_image(cell, cutoff):
    ncutoff = cutoff + 1e-8 * cutoff
    V = np.abs(np.linalg.det(cell))
    a, b, c = cell
    Xa = np.cross(b, c)
    ha = V / np.linalg.norm(Xa)
    na = int(np.ceil(2 * ncutoff / ha))
    Xb = np.cross(a, c)
    hb = V / np.linalg.norm(Xb)
    nb = int(np.ceil(2 * ncutoff / hb))
    Xc = np.cross(a, b)
    hc = V / np.linalg.norm(Xc)
    nc = int(np.ceil(2 * ncutoff / hc))
    return na, nb, nc


@dataclass(frozen=True, slots=True)
class Adsorption(ami.abc.CalculatorInterface):
    workdir: Path

    force_field: str
    force_field_mixing_rules: str
    pseudo_atoms: str
    SO2: str
    CO2: str
    N2: str
    input_template: str

    @classmethod
    def from_template_folder(cls, workdir: Union[str, Path], path: Union[str, Path]):
        print()
        path = Path(path)
        data = {}
        for f in fields(cls):
            if f.name == "workdir":
                continue
            data[f.name] = (path / f'{f.name}.def').read_text("utf8")
        return cls(workdir=Path(workdir), **data)

    def write(self, cif_bytes: bytes, subdir: str):
        w = Path(self.workdir)/subdir
        w.mkdir(parents=True, exist_ok=True)
        atoms = read(BytesIO(cif_bytes), format="cif")
        cell = np.array(atoms.cell)

        cutoff = 12.0
        na, nb, nc = find_minimum_image(cell, cutoff)

        for f in fields(self.__class__):
            name = f.name
            if (name != "input_template") and (name != "workdir"):
                data = getattr(self, name)
                (w / f'{name}.def').write_text(data)
            else:
                tpl = self.input_template
                data = tpl.format(cutoff=cutoff, na=na, nb=nb, nc=nc)
                (w / "simulation.input").write_text(data)

        (w / "simulation.cif").write_bytes(cif_bytes)

        # Remove existing data if relevant
        for out_path in w.glob("Output/System_0/*.data"):
            out_path.unlink()

    def run_external(self, subdir: str):
        run(["simulate", "simulation.input"], cwd=self.workdir/subdir)

    def parse_output(self, subdir: str):
        ads_pressure="100000"            # Define pressure manually for Wiersum API calculation
        des_pressure="10000"
    
        ads_path = list((self.workdir/subdir).glob(f"Output/System_0/*{ads_pressure}.data"))[0]
        des_path = list((self.workdir/subdir).glob(f"Output/System_0/*{des_pressure}.data"))[0]
  
        ads_loading = {}
        des_loading = {}
        ads_enthalpy = {}
        des_enthalpy = {}
        
        #Collect enthalpies of components
        with ads_path.open(mode="r") as fd:
            for line in fd:
                if "Enthalpy of adsorption:" in line:
                    break
            for line in fd:
                if line.startswith("	Enthalpy of adsorption component"):
                    name = line.split()[-1][1:-1]                   # Extract the component name
                if line.strip().endswith("[KJ/MOL]") and name not in ads_enthalpy:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the enthalpy in KJ/MOL
                    ads_enthalpy[name] = res
                    
        #Collect loading of components       
        with ads_path.open(mode="r") as fd:
            for line in fd:
                if "Number of molecules:" in line:
                    break
            for line in fd:
                if line.startswith("Component"):
                    name = line.split()[-1][1:-1]                   # Extract the component name
                if "Average loading excess [cm^3 (STP)/cm^3 framework]" in line:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the volumetric uptake [cm3/cm3]
                    ads_loading[name] = res     
                    
        with des_path.open(mode="r") as fd:
            for line in fd:
                if "Enthalpy of adsorption:" in line:
                    break
            for line in fd:
                if line.startswith("	Enthalpy of adsorption component"):
                    name = line.split()[-1][1:-1]                   # Extract the component name
                if line.strip().endswith("[KJ/MOL]") and name not in des_enthalpy:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the enthalpy in KJ/MOL
                    des_enthalpy[name] = res
                    
        with des_path.open(mode="r") as fd:
            for line in fd:
                if "Number of molecules:" in line:
                    break
            for line in fd:
                if line.startswith("Component"):
                    name = line.split()[-1][1:-1]                   # Extract the component name
                if "Average loading excess [cm^3 (STP)/cm^3 framework]" in line:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the volumetric uptake [cm3/cm3]
                    des_loading[name] = res                         # Store the result in the dictionary
                
        return ads_loading, des_loading, ads_enthalpy, des_enthalpy
        
    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]
        self.write(cif_bytes, subdir=subdir)
        self.run_external(subdir=subdir)                
        ads_loading, des_loading, ads_enthalpy, des_enthalpy = self.parse_output(subdir=subdir)
        
        working_capacity = ads_loading["SO2"] - des_loading["SO2"]
        enthalpy_change = ads_enthalpy["SO2"] - des_enthalpy["SO2"]
        selectivity = (ads_loading["SO2"] / ads_loading["CO2"]) / (0.002/0.198)
        API = (selectivity - 1) * working_capacity / abs(enthalpy_change)    # abs() returns absolute value, will alway be positive
        
        return np.log(API)

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('cif_content', bytes), ('subdir', str)],
            output_schema=[('API', float)]
        )
