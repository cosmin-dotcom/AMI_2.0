from dataclasses import dataclass, fields, field
from io import BytesIO
from pathlib import Path
import shutil
from subprocess import run
from typing import Union, List

import numpy as np
from ase.io import read

import ami.abc
from ami.abc import SchemaInterface
from ami.schema import Schema
from ami.serialized_opaque import SerializedOpaque

import breakthrough_model
import matplotlib.pyplot as plt

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

# # --------------------------------------------------------------------------------------------------------

# These should be all the parameters you ever need to change

gas_components = ["H2","CO","N2","CO2","CH4"]
gas_fractions = [0.073,0.004,0.004,0.016,0.003] # sum(gas_fractions) should be <= 0.1, as carrier gas will be >=0.9
pressure_range = [1e3,1e4,4e4,7e4,1e5,4e5,7e5,1e6,4e6,7e6,1e7,1e8]
pressure_range = ' '.join(map(str, pressure_range))

# # --------------------------------------------------------------------------------------------------------

@dataclass(frozen=False, slots=True)
class Adsorption(ami.abc.CalculatorInterface):
    workdir: Path
    force_field: str
    force_field_mixing_rules: str
    pseudo_atoms: str
    input_gcmc: str
    # We set a list of gas components as an attribute so that it gets carried through all class methods
    # It allows you to define the number of components once, and the rest is done automatically
    gas_components: list = field(default_factory=lambda: gas_components) 

    @classmethod
    def from_template_folder(cls, workdir: Union[str, Path], path: Union[str, Path]):
        path = Path(path)
        data = {}

        # Resolve gas_components from field defaults
        gas_components = None
        for f in fields(cls):
            if f.name == "gas_components":
                gas_components = f.default_factory()  # call the factory to get the list
                break
        
        for f in fields(cls):
            if f.name == "workdir":
                continue
            if f.name == "gas_components":
                # gas_components is an attribute, and data is a new dictionary of attributes
                # So each gas should not be accessed from data, but from a dict in data
                data["gas_components"] = {component: (path / f'{component}.def').read_text("utf8") for component in gas_components}
            else:
                data[f.name] = (path / f'{f.name}.def').read_text("utf8")
        
        return cls(workdir=Path(workdir), **data)
    
    def write(self, cif_bytes: bytes, subdir: str):
        w = Path(self.workdir)/subdir
        w.mkdir(parents=True, exist_ok=True)
        atoms = read(BytesIO(cif_bytes), format="cif")
        cell = np.array(atoms.cell)

        cutoff = 13.0
        na, nb, nc = find_minimum_image(cell, cutoff)

        # Write input GCMC

        gas_components = self.gas_components # This actually loads the dictionary with the def file bytes

        for f in fields(self.__class__):
            name = f.name
            if f.name == "workdir":
                continue
            elif name == "gas_components":
                print(type(gas_components))
                for key, value in gas_components.items():
                    (w / f'{key}.def').write_text(value)
            elif (name != "input_gcmc"):
                data = getattr(self, name)
                (w / f'{name}.def').write_text(data)
            else:
                template = self.input_gcmc
                for component in gas_components:
                    data = template.format(cutoff=cutoff, na=na, nb=nb, nc=nc, pressure_range=pressure_range, component=component)
                    (w / f"{component}_isotherm.gcmc").write_text(data)       
                
            
        # Write the CIF for the COF we are using
        (w / "simulation.cif").write_bytes(cif_bytes)

        # Remove existing data if relevant
        for out_path in w.glob("Output/System_0/*.data"):
            out_path.unlink()
       
    def run_external(self, subdir: str):                                   
        for component in self.gas_components:
            run(["simulate", "-i", f"{component}_isotherm.gcmc", "-a", f"_.{component}"], cwd=self.workdir/subdir)
        # This will create output files in the form: *_{pressure}_.{component}.data
        # Example: output_simulation_4.3.3_298.000000_7e+06_.N2.data
    
    def parse_output(self, subdir: str):
        # Initiate dictionary with empty lists for components
        ads_loading = {key: [] for key in self.gas_components}
            
        ads_path = list((self.workdir/subdir).glob(f"Output/System_0/*.data"))
        
        for file in ads_path:
            #grab component
            component_name = str(file).split('.')[-2]
            #grab pressure
            pressure_point = str(file).split('_')[-2]
            #grab results    
            with file.open(mode="r") as fd:
                for line in fd:
                    if "Number of molecules:" in line:
                        break
                for line in fd:
                    if line.startswith("Component"):
                        name = line.split()[-1][1:-1]
                        assert name == component_name, f"output.data file does not match"
                        # Extract the component name
                    if "Average loading absolute [mol/kg framework]" in line:
                        res = line.split(" +/-")[0].split()[-1] # Extract the volumetric uptake [cm3/cm3]
                        p_ads_tuple = (float(pressure_point),float(res)) #Tuple of floats
                        ads_loading[component_name].append(p_ads_tuple)

        return ads_loading
        
    def calculate(self, parameters: SerializedOpaque) -> SerializedOpaque:
        subdir = parameters["subdir"]
        cif_bytes = parameters["cif_content"]
        self.write(cif_bytes, subdir=subdir)
        self.run_external(subdir=subdir)                

        raw_isotherm_data = self.parse_output(subdir=subdir) # a list of tuples
        print(raw_isotherm_data)
        
        with open(f"{self.workdir}/{subdir}/output_{subdir}.txt", 'wt') as out:
            out.write(str(raw_isotherm_data))

        print("Starting isotherm fitting.")
        gas_components=self.gas_components
        breakthrough_input = {}
        model="Langmuir"
        num_params=2
        sites=2
        
        one_site_isotherm = [model]
        [one_site_isotherm.append(float(0.0)) for i in range(num_params)]

        # Initite pyplot for isotherms
        fig, ax = plt.subplots(figsize=(12,6))

        # Start fitting each isotherm independently 
        # So that you can re-run once if it fails without re-running all
        # Also for analysing individual R2 and parameter values. It's just nicer and controlled :)
        for component in gas_components:
            attempts = 0
            fitted = False
            while attempts < 3:
                unfitted_object = breakthrough_model.Component_Object(isotherm_name=component,
                                                                    raw_adsorption_data=raw_isotherm_data[component],
                                                                    num_params=num_params,
                                                                    sites=sites,
                                                                    model=model
                                                                    ) 

                params, r2 = unfitted_object.fit_component_to_model(savepath=f"{self.workdir}/{subdir}",
                                                                    ax_for_pyplot=ax)

                #Checks for good isotherm
                if r2 < 0.9 or np.isnan(r2):
                    print(f"R2={r2} criteria not met, or got a NaN value!")
                    attempts+=1
                elif params[0][1] > 1000 or params[1][1] > 1000:
                    print(f"Component {component} got a huge Q_max value so did not fit properly")
                    attempts+=1
                else:
                    print("FITTING SUCCESSFUL")
                    fitted = True
                    break
            
            if not fitted:
                print(f"WARNING: Component {component} failed to fit after 3 attempts")
            
            # Assign only the isotherm parameters which pass the checks
            breakthrough_input[f"{component}"] = params # Should be list[str, float, float...]*n
        
        # Plot graph after everything is fitted
        ax.legend()
        plt.savefig(f"{self.workdir}/{subdir}/ALL_isotherms.png")
        plt.close()

        # Tidy up large Output folder, but save data if we want to run separately later
        with open(f"{self.workdir}/{subdir}/output_{subdir}.txt", 'a') as out:
            out.write(str(breakthrough_input))
        out.close()
        dir_path = Path(self.workdir) / subdir / "Output"
        shutil.rmtree(dir_path)

        # Start breakthrough
        brk_object = breakthrough_model.Breakthrough_Object()
        print(f"Starting breakthrough for MOF {subdir}")
        brk_object.ruptura_breakthrough(gas_components=gas_components,
                                      gas_fractions=gas_fractions,
                                      params_dict=breakthrough_input,
                                      savepath=f"{self.workdir}/{subdir}"
                                      )
        
        print("BREAKTHROUGH SUCCESSFUL")

        # Calculate purity time of desired component
        
        try:
            purity_time, time, purity, time_max_purity = brk_object.find_H2_purity(brk_object.data, 
                                                                  gas_components, 
                                                                  brk_object.time_interval, 
                                                                  savepath=f"{self.workdir}/{subdir}")
            brk_object.plot_purity(time, 
                                   purity, 
                                   savepath=f"{self.workdir}/{subdir}/python_purity.png")
        except Exception as e:
            print(e)
        
        target = np.log(purity_time+1) # Makes sure we never run log(0)
        
        return {
            'target': target,  # This is what the surrogate will use
            'purity_time[s]': purity_time,
            'time_max_purity[s]': time_max_purity
            }

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('cif_content', bytes), ('subdir', str)],
            output_schema=[
                ('target', float)
                ]
        )
