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
    xenon: str
    krypton: str
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

        cutoff = 16.0
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
        base_path = list((self.workdir/subdir).glob("Output/System_0/*.data"))[0]

        components = {}
        with base_path.open(mode="r") as fd:
            for line in fd:
                if "Number of molecules:" in line:
                    break
            for line in fd:
                if line.startswith("Component"):
                    name = line.split()[-1][1:-1]
                if "Average loading absolute   " in line:
                    res = float(line.split(" +/-")[0].split()[-1])
                    components[name] = res
        return components

    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]
        self.write(cif_bytes, subdir=subdir)
        self.run_external(subdir=subdir)
        components = self.parse_output(subdir=subdir)
        absorbed_Xe = components["xenon"]
        absorbed_Kr = components["krypton"]
        return np.log(1 + (4 * absorbed_Xe)) - np.log(1 + absorbed_Kr)

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('cif_content', bytes), ('subdir', str)],
            output_schema=[('selectivity', float)]
        )
