from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Union

import ami.abc
from ami.abc import SchemaInterface
from ami.schema import Schema
from ami.serialized_opaque import SerializedOpaque

import traceback

@dataclass(frozen=True, slots=True)
class Simulation(ami.abc.CalculatorInterface):
    workdir: Path

    @classmethod
    def run_external(self, subdir: str):
        """run is a python module from the subprocesses library which can send jobs to your simulation program"""
        run(["simulate", "simulation.input"], cwd=self.workdir/subdir) 

    def parse_output(self, subdir: str):
        """Reads output files from simulation program and returns data for the calculator"""
        base_path = list((self.workdir/subdir).glob("where_your_output_files_are"))[0]
        return parsed_data
        
    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        """
        This is the public calculate() method that the AMI machinery sees.
        You must edit the helper calculate method below where the target is defined!
        Why: because the public method never use to raise Exceptions so I did not know what was crashing.
        """
        try:
            return self._calculate_help(parameters)
        except Exception:
            traceback.print_exc() # ami.mp.runner() will catch this too
            raise

    def _calculate_help(self, parameters: SerializedOpaque) -> SerializedOpaque:
        """Transforms the data from parse_output into a target for the AMI to """
        subdir = parameters["subdir"]
        file_bytes = parameters["file_content"]
        self.write(file_bytes, subdir=subdir)
        self.run_external(subdir=subdir)
        parsed_data = self.parse_output(subdir=subdir)
        
        return target

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('file_content', bytes), ('subdir', str)],
            output_schema=[('target', float)]
        )
