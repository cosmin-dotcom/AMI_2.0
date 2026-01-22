from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Union

import numpy as np
from ase.io import read

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

    force_field: str
    force_field_mixing_rules: str
    pseudo_atoms: str
    D4: str
    Tip5p: str
    input_cfcmc: str
    input_widom: str

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

        cutoff = 13.0
        na, nb, nc = find_minimum_image(cell, cutoff)

        #Write input CFCMC

        for f in fields(self.__class__):
            name = f.name
            if (name != "input_cfcmc") and (name != "workdir"):
                data = getattr(self, name)
                (w / f'{name}.def').write_text(data)
            else:
                tpl = self.input_cfcmc
                data = tpl.format(cutoff=cutoff, na=na, nb=nb, nc=nc)
                (w / "simulation.cfcmc").write_text(data)

        #Write input WIDOM

        for f in fields(self.__class__):
            name = f.name
            if (name != "input_widom") and (name != "workdir"):
                data = getattr(self, name)
                (w / f'{name}.def').write_text(data)
            else:
                tpl = self.input_widom
                data = tpl.format(cutoff=cutoff, na=na, nb=nb, nc=nc)
                (w / "simulation.widom").write_text(data)       

        #Write the CIF for the COF we are using

        (w / "simulation.cif").write_bytes(cif_bytes)

        # Remove existing data if relevant
        for out_path in w.glob("Output/System_0/*.data"):
            out_path.unlink()
       
    def run_external(self, subdir: str):                                   
        run(["simulate", "-i", "simulation.widom", "-a", "_widom"], cwd=self.workdir/subdir)
        run(["simulate", "-i", "simulation.cfcmc", "-a", "_cfcmc"], cwd=self.workdir/subdir)

    def parse_output(self, subdir: str):
        
        ads_path = list((self.workdir/subdir).glob(f"Output/System_0/*cfcmc.data"))[0]
        henry_path = list((self.workdir/subdir).glob(f"Output/System_0/*widom.data"))[0]
        
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
                name = "D4"                 # since it is single-component, there is no specific mention of Component names in output so we have to specify
                if line.strip().endswith("[KJ/MOL]") and name not in ads_enthalpy:
                    print(line)
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the enthalpy in KJ/MOL
                    ads_enthalpy[name] = res
                    print(res)
                    
        #Collect loading of D4     
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
                    
        #Collect Henry of Tip5 
        with henry_path.open(mode="r") as fd:
            for line in fd:
                if "Average Henry coefficient:" in line:
                    break
            for line in fd:
                if "] Average Henry coefficient:" in line:
                    name = line.split()[0][1:-1]                    # Extract the component name
                    res = float(line.split(" +/-")[0].split()[-1])  # Extract the Average Henry [mol/kg/Pa]
                    ads_henry[name] = res      

        return ads_loading, ads_enthalpy, ads_henry
        
    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]
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