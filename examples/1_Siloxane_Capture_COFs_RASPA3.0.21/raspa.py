from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Union

import os
import shutil
import re
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
    D4: str
    Tip5p: str
    raspa3_executable: str
    force_field: str
    input_cfcmc: str
    input_widom: str

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

        #FORMAT input_cfcmc.json to RASPA3 simulation.json
        for f in fields(self.__class__):
            name = f.name
            # First write the {component.json} files, including the executable raspa3_executable.json
            # executable must have executing permissions!
            if name == "workdir":
                continue
            if (name != "workdir") and (name != "force_field"):
                data = getattr(self, name)
                (w / f'{name}.json').write_text(data)
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

        #Write the CIF for the COF we are using
        (w / "COF.cif").write_bytes(cif_bytes)
        
        # Remove existing data if relevant
        for out_path in w.glob("Output/System_0/*.data"):
            out_path.unlink()
       
    def run_external(self, subdir: str):
        print("Running Widom")                                   
        
        run(["chmod","+x","raspa3_executable.json"], cwd=self.workdir/subdir) # make sure we can run it
        shutil.copy('./input_widom.json', './simulation.json')                # make simulation input have correct name
        
        run(["./raspa3_executable.json"], cwd=self.workdir/subdir)            # RUN WIDOM
        print("WIDOM SUCCESS")
        
        print("Running CFCMC")
        
        # Read original
        with open("force_field.json") as f:
            data = json.load(f)

        # Modify
        data["ChargeMethod"] = "None"

        # Write temp file
        with open("force_field_no_charge.json", "w") as f:
            json.dump(data, f)
            
        shutil.copy('./input_cfcmc.json', './simulation.json')                # make cfcmc input have correct name
        run(["./raspa3_executable.json"], cwd=self.workdir/subdir)
        print("CFCMC SUCCESS")
 
    def parse_output(self, subdir: str):
        
        ads_path = list((self.workdir/subdir).glob(f"output/*_10.s0.txt"))[0]
        henry_path = list((self.workdir/subdir).glob(f"output/*_0.s0.txt"))[0]
        
        assert ads_path.exists(), f"output file not found: {p}"
        assert henry_path.exists(), f"output file not found: {p}"

        ads_loading = {}
        ads_enthalpy = {}
        ads_henry = {}

        #Collect enthalpies of D4
        with ads_path.open(mode="r") as fd:
            for line in fd:
                if "Enthalpy of adsorption:" in line:
                    break
            for line in fd: 
                name = "D4"
                if line.strip().endswith("[kJ/mol]") and name not in ads_enthalpy:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the enthalpy in KJ/MOL
                    ads_enthalpy[name] = res
                    print(res)
                    
        #Collect loading of D4     
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
                    
        #Collect Henry of Tip5 
        with henry_path.open(mode="r") as fd:
            for line in fd:
                if "Average Henry coefficient:" in line and "[mol/kg/Pa]" in line:
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the Average Henry [mol/kg/Pa]
                    ads_henry["Tip5p"] = res    # I got lazy automating this for multiple components, sorry!  

        return ads_loading, ads_enthalpy, ads_henry
        
    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]
        
        # Fix invalid CIF label before anything reads/writes it
        cif_bytes = re.sub(rb"_atom_site_label", rb"_atom_site.id", cif_bytes)
        self.write(cif_bytes, subdir=subdir)

        self.run_external(subdir=subdir)
        
        ads_loading, ads_enthalpy, ads_henry = self.parse_output(subdir=subdir)
        
        target = np.log(1/ads_henry["Tip5p"]) + np.log(ads_loading["D4"])
        
        return {
            'target': target,  # This is what the surrogate will use
            'ads_loading_D4': ads_loading["D4"],
            'D4_enthalpy': ads_enthalpy["D4"],
            'henry_Tip5p': ads_henry["Tip5p"]
        }

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('cif_content', bytes), ('subdir', str)],
            output_schema=[('log(target)', float),
            ('D4 ads load [cm3/cm3]', float), 
            ('D4 ads enthalpy [kJ/mol]', float),
            ('Tip5p Henry constant [mol/kg/Pa]', float)]
        )
