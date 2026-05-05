from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Union

import numpy as np
from ase.io import read
import json 

import ami.abc
from ami.abc import SchemaInterface
from ami.schema import Schema

from ami.serialized_opaque import SerializedOpaque

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
    H2: str
    CO2: str
    CO: str
    N2: str
    CH4: str
    raspa3_executable: str
    force_field: str
    input_gcmc: str

    @classmethod
    def from_template_folder(cls, workdir: Union[str, Path], path: Union[str, Path]):
        path = Path(path)
        data = {}

        for f in fields(cls):
            if f.name == "workdir":
                continue
            else:
                print(f.name)
                data[f.name] = (path / f'{f.name}.json').read_text("utf8")
        print("Grabbed all JSON info")

        return cls(workdir=Path(workdir), **data)

    def write(self, cif_bytes: bytes, subdir: str):
        w = Path(self.workdir)/subdir
        w.mkdir(parents=True, exist_ok=True)
        atoms = read(BytesIO(cif_bytes), format="cif")
        cell = np.array(atoms.cell)

        cutoff = 13.0
        na, nb, nc = find_minimum_image(cell, cutoff)

        #FORMAT input_gcmc.json to RASPA3 simulation.json
        for f in fields(self.__class__):
            print(f"Currently writing: {f.name}, type: {type(f.name)}")
            name = f.name
            # First write the {component.json} files, including the executable raspa3_executable.json
            # executable must have executing permissions!
            if name == "workdir":
                continue
            if (name != "input_gcmc") and (name != "workdir") and (name != "force_field"):
                print("Entered first if statement")
                data = getattr(self, name)
                (w / f'{name}.json').write_text(data)
                print(f"wrote {name} data")
            elif name == "force_field":
                # Parse the JSON data
                data = json.loads(getattr(self, name))

                # Specify the field key to update
                field_key = "CutOffVDW"
    
                # Update the specified field value
                if field_key in data:
                    data[field_key] = cutoff

                # Convert the modified data back to JSON
                modified_json = json.dumps(data)
                (w / f"force_field.json").write_text(modified_json)
            else:
                # Parse the JSON data
                data = json.loads(getattr(self, name))
                
                # Specify the field key to update
                field_key = "NumberOfUnitCells"

                # Update the specified field value
                if field_key in data["Systems"][0]:
                    data["Systems"][0][field_key] = [na, nb, nc]
                else:
                    print(f"ERROR: {field_key} not found in data. Available keys: {list(data.keys())}") 
                    data[field_key] = [na, nb, nc]
                    print(f"After assignment, data['{field_key}']: {data[field_key]}")

                # Convert the modified data back to JSON
                modified_json = json.dumps(data)
                (w / f"simulation.json").write_text(modified_json) 

        #Write the CIF for the COF we are using
        (w / "simulation.cif").write_bytes(cif_bytes)
 
    def run_external(self, subdir: str):                                   
        run(["chmod","+x","raspa3_executable.json"], cwd=self.workdir/subdir)
        run(["./raspa3_executable.json"], cwd=self.workdir/subdir)

    def parse_output(self, subdir: str):
        ads_path = list((self.workdir/subdir).glob(f"output/*.txt"))[0]
        assert ads_path.exists(), f"output file not found: {p}"

        ads_loading = {}
        
        with ads_path.open(mode="r") as fd:
            for line in fd:
                if "Loadings" in line:
                    break
            for line in fd:
                if line.startswith("Component"):
                    name = line.split()[-1][1:-1]                   # Extract the component name
                if "Abs. loading average" in line and "[mol/kg-framework]" in line:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the mol/kg uptake
                    ads_loading[name] = res               

        return ads_loading

    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]

        self.write(cif_bytes, subdir=subdir)
        
        self.run_external(subdir=subdir)                
        
        ads_loading = self.parse_output(subdir=subdir)
        
        N2_load = ads_loading["N2"]
        CO_load = ads_loading["CO"]
        H2_load = ads_loading["H2"]

        impurities = (N2_load+CO_load)/0.08 # Divided by total mol frac
        desired = H2_load/0.73

        selectivity = impurities/(desired+1)
        capacity_impurities = N2_load+CO_load
        target = np.log(capacity_impurities) + np.log(selectivity)
        
        return {
            'target': target,  # This is what the surrogate will use
            'selectivity (CO+N2/H2) [-]': selectivity,
            'CO_N2_capacity': capacity_impurities,
            'load_CO2': ads_loading["CO2"],
            'load_H2': ads_loading["H2"],
            'load_N2': ads_loading["N2"],
            'load_CO': ads_loading["CO"],
            'load_CH4': ads_loading["CH4"]
        }

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('cif_content', bytes), ('subdir', str)],
            output_schema=[
                ('target', float),
                ('selectivity (CO+N2/H2) [-]', float),
                ('CO+N2 Capacity [cm3/cm3]', float),
                ('CO2 loading [cm3/cm3]', float),
                ('H2 loading [cm3/cm3]', float),
                ('N2 loading [cm3/cm3]', float),
                ('CO loading [cm3/cm3]', float),
                ('CH4 loading [cm3/cm3]', float)
                ]
        )
