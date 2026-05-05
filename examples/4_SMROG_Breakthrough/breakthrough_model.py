import ruptura
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import ctypes
import pandas # To test and save purity calculations

class CaptureCStdout:
    def __enter__(self):
        self._tmpfile = tempfile.TemporaryFile(mode='w+')
        self._old_fd = os.dup(1)
        os.dup2(self._tmpfile.fileno(), 1)
        return self

    def __exit__(self, *args):
        # Flush C-level stdout before restoring
        libc = ctypes.CDLL(None)
        libc.fflush(None)  # fflush(NULL) flushes all open C streams
        os.dup2(self._old_fd, 1)
        os.close(self._old_fd)
        self._tmpfile.seek(0)
        self.lines = self._tmpfile.readlines()
        self._tmpfile.close()

class Component_Object():
    isotherm_params: list[float]
    r2: float

    def __init__(self, raw_adsorption_data: list[tuple], 
                 isotherm_name: str, 
                 num_params: int, 
                 sites: int, 
                 model: str = "Langmuir"
                 ):
        self.raw_adsorption_data = [raw_adsorption_data]
        self.isotherm_name = isotherm_name
        self.model = model
        self.sites = sites
        self.num_params = num_params
    
    def fit_component_to_model(self, savepath, ax_for_pyplot):
        """
        Takes a pure component isotherm and returns the model used and its parameters.
        savepath is for graphing and saving figure in appropriate folder.
        ax_for_pyplot is a fig, ax created outside of this script. When fitting.plot is called, the isotherm model graph is appened to that ax object.
        """

        one_site_isotherm = [self.model]
        [one_site_isotherm.append(float(0.0)) for i in range(self.num_params)]
        components = ruptura.Components()

        components.addComponent(
            MoleculeName = f"{self.isotherm_name}",
            GasPhaseMolFraction=1.0,
            isotherms = [one_site_isotherm]*self.sites #list[lists]
        )

        fitting = ruptura.Fitting(
            components=components,
            DisplayName=f"{savepath}"
        )
        
        print(f"[breakthrough_model] Starting fitting for {self.isotherm_name} with {self.model} model.")
        with CaptureCStdout() as cap:
            results = fitting.compute(self.raw_adsorption_data)
        
        # cap.lines is a list of strings, one per line
        r2 = [l.strip().split(' ')[-1] for l in cap.lines if "R2" in l] # Returns a list[str]
        
        self.isotherm_params = results # This is just a list of floats
        self.r2 = float(r2[0]) # Turns list[str] into float

        # -----------------------------------------------------------------------------------
        try:
            p = np.logspace(0, 10, 200)
            # [0, 7] is a list of ruptura.fitting.plot() possible axis titles. 
            # We choose Pa and mol/kg as this is the data collected from RASPA.
            fitting.plot(ax_for_pyplot, self.raw_adsorption_data, p, *[0,7])
        except Exception as e:
            print(f"Error plotting isotherm in breakthrough_model:{e}")
        # -----------------------------------------------------------------------------------
        
        # We actually need to return [str, float, float...] input for the breakthrough object below... 
        
        brk_input = [] # list[lists]
        
        # For multi-site, we need to handle multiple parameters as separate inputs for breakthrough
        for site in range(self.sites):
            input_line = [f"{self.model}"]
            [input_line.append(float(results[site*self.num_params+i])) for i in range(self.num_params)] 
            brk_input.append(input_line)

        return brk_input, self.r2

class Breakthrough_Object():
    data: str
    time_interval: float   

    def __init__(self):
        print("Initialised a Breakthrough_Object")

    def get_integral(self, y_array, trapezium_base):
        # to do the sum of the parallel sides, we have to get another array that is one index ahead
        # but starts with 0 and ends with the second to last value 
        y_array_shifted = y_array
        y_array_shifted[1:len(y_array)] = y_array[0:len(y_array)-1]
        #y_array_shifted[0] = 0                           This part, although correct, will always result in a division failure. We correct later
        result = 0.5*trapezium_base*(y_array+y_array_shifted) # Trapezium rule
        return result
    
    def plot_purity(self, time, purity, savepath: str):
        fig, ax1 = plt.subplots(1, 1)
        scatter1 = ax1.scatter(time, purity, color='#408e1c', label=f"H2 purity, {len(purity)} samples")
        ax1.set_xlabel('Time [min]')
        ax1.set_ylabel('Purity [%]')
        ax1.set_ylim([0, 100])
        ax1.legend()
        plt.savefig(savepath, dpi=100)
        plt.close()

    def find_H2_purity(self, brk_data, gas_components, time_interval, savepath):
        # Index 0: z  (column position)
        # Index 1: V  (velocity)
        # Index 2: z  (column position)
        # Index 3: V  (velocity)
        # Index 4: Pt (total pressure)
        # Index 5: component 0 Q     (loading)                     ---Helium, carrier
        # Index 6: component 0 Qeq   (equilibrium loading)
        # Index 7: component 0 P     (partial pressure)
        # Index 8: component 0 Pnorm (normalized partial pressure)
        # Index 9: component 0 Dpdt  (derivative P with t)
        # Index 10: component 0 Dqdt  (derivative Q with t)
        # Index 11: component 1 Q     (loading)                     ---Hydrogen
        # Index 12: component 1 Qeq   (equilibrium loading)
        # Index 13: component 1 P     (partial pressure)
        # Index 14: component 1 Pnorm (normalized partial pressure)
        # Index 15: component 1 Dpdt  (derivative P with t)
        # Index 16: component 1 Dqdt  (derivative Q with t)
        # Index 17: component 2 Q     (loading)                     ---Next component
        # Index 18: component 2 Qeq   (equilibrium loading)
        # Index 19: component 2 P     (partial pressure)
        # Index 20: component 2 Pnorm (normalized partial pressure)
        # Index 21: component 2 Dpdt  (derivative P with t)
        # Index 22: component 2 Dqdt  (derivative Q with t)
            
        # brk_data is a 3D array in the shape of (time_intervals, grid points, loading_data)

        effluent_dict = {key: np.array([]) for key in gas_components}
        
        velocity_at_col_end = np.array([])
        
        for i in range(len(brk_data)):
            # The range is 0 ---> 167, len=168
            one_step = brk_data[i]
            at_col_end = one_step[-1] # The last one is always the final length, final grid point
            for i, component in enumerate(gas_components, start=1):
                # For x components, loading is 6x+5, excluding Carrier gas
                # i will start at 0, so we add enumerate(start=1)
                effluent_dict[component] = np.append(effluent_dict[component], at_col_end[6*i+5])
            
            # Let's also add the velocity at the end of the column
            velocity_at_col_end = np.append(velocity_at_col_end, at_col_end[3]) # idx 1 is the velocity in ruputura output

        # Add numerators together (int) for total saturation in column at the end
        desired = effluent_dict['H2']*velocity_at_col_end
        total = sum([effluent_dict[component]*velocity_at_col_end for component in gas_components])
        numerator = self.get_integral(desired[1:], trapezium_base=time_interval)
        denominator = self.get_integral(total[1:], trapezium_base=time_interval)
        # For the first 1.6 seconds, it seems nothing comes through at all, hence denominator is 0! Error!
        # We can handle it using np.divide with 'where' argument, where 
        purity = np.divide(numerator*100, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator!=0)

        time = [] # in seconds
        [time.append(i*time_interval) for i in range(0,len(purity[0:]))] # says, "give me one time step for every data point"
        time = np.array(time)/60 # in minutes

        target_purity=99.0
        
        # Assumptions and observations:
        # 1. Sometimes purity seems to come out low, then increase to a max, then lower again.
        # 2. Always expect purity to drop within 300 seconds of breakthrough simulation, given 99.0+ target purity
        #    Hence we stop breakthrough simulation after 300 seconds to save comp. resources

        idx_max_purity = np.argmax(purity) # Gives index of first occurence of highest value
        time_max_purity = idx_max_purity*time_interval
        for index, percentage in enumerate(purity):
            if index < idx_max_purity:
                continue
            if float(percentage) < target_purity:
                threshold_purity = index*time_interval
                threshold_min = threshold_purity/60
                break
            
        print(f"{target_purity}% H2 purity up until {threshold_purity} [s] ({threshold_min} [min])")
        print(f"For {len(purity)} data points.")
        
        purity_dict = {'H2 C/C0':effluent_dict['H2'],
                               'Velocity recorded at end of column [m/s]':velocity_at_col_end,
                               'H2*velocity':desired,
                               'Integral of H2*velocity':numerator,
                               'Integral of Impurities':denominator,
                               'purity [%]':purity
                               }
                               
        df = pandas.DataFrame(dict([(key, pandas.Series(value)) for key, value in purity_dict.items()]))
                               
        df.to_csv(f"{savepath}/PURITY_CALCULATIONS.csv")
            
        return threshold_purity, time, purity, time_max_purity

    def ruptura_breakthrough(self, gas_components, gas_fractions, params_dict, savepath):
        print("Entering brk script()")
        print(params_dict)

        list_of_dicts = []
        helium_carrier = {
            "MoleculeName": "Helium",
            "CarrierGas": True,
            "GasPhaseMolFraction": 1-sum(gas_fractions),
            "isotherms": []
            }
        
        list_of_dicts.append(helium_carrier)

        for component, fraction in zip(gas_components, gas_fractions):
            list_of_dicts.append({   
            "MoleculeName": f"{component}",
            "GasPhaseMolFraction": fraction,
            "MassTransferCoefficient": 0.06,
            "AxialDispersionCoefficient": 0.0,
            "isotherms": params_dict[component] # Should already be a list of list(s) from fit method
            })
        
        ruptura_components = ruptura.Components(list_of_dicts)
        
        TimeStep=5e-4 # Seconds
        WriteEvery=200
        self.time_interval=TimeStep*WriteEvery
        stop_brk_at=200 # Seconds
        num_time_steps=stop_brk_at/TimeStep

        brk = ruptura.Breakthrough(
            components=ruptura_components,
            DisplayName=f"{savepath}",
            Temperature=298.0,
            ParticleDensity=1126.7,
            TotalPressure=7e6,
            PressureGradient=0.0,
            ColumnEntranceVelocity=0.1,
            ColumnLength=0.3,
            NumberOfTimeSteps=num_time_steps,
            TimeStep=TimeStep,
            NumberOfGridPoints=100,
            PrintEvery=WriteEvery,
            WriteEvery=WriteEvery,
            PulseTime=None,
            MixturePredictionMethod="IAST",
        )
        
        print("[breakthrough_model] Starting breakthrough...")        
        with CaptureCStdout() as cap:
            data = brk.compute()
        try:
            fig, ax = plt.subplots(figsize=(8,6))
            brk.plot(ax, "breakthrough")
            plt.savefig(f'{savepath}/breakthrough_graph.png', dpi=100)
            plt.close()
        except Exception as e:
            print(f"Error plotting breakthrough graph:{e}")
        
        self.data = data
        return data
