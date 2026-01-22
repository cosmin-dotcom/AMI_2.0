#!/bin/sh
#Script for getting loading data from binary simulation: AT ONE TEMPERATURE


echo "Finding loadings..."

#TO DO IN FUTURE: use the selectivity from binary simulations, not a constant Hc selectivity.

#What conditions are adsorption and desorption ocurring at?
pressure_ads=100000
pressure_des=10000

#touch API.dat

echo "Dir,SO2[mg/g],SO2_des[mg/g],CO2[mg/g],working_capacity[mg/g],enthalpy_ads[kJ/mole],enthalpy_des[kJ/mole],change_enthalpy[kJ/mole],API" >> API.dat

for dir in `ls -d */`
do
        cd $dir/Output/System*/ 
        
        
        #FIRST WE FIND THE LOADING OF COMPONENTS (no units)
        #Ouput from grep should look like Average loading absolute [cm^3 (STP)/cm^3 framework] 333.4183536218 +/- 3.1781516675 [-]
        
        #We use the volumetric measurement because the loading depends on this!
        
        #Finding the working capacity
       	uptakeSO2=$( grep "cm^3 (STP)/cm^3 framework" *"$pressure_ads".data | head -3 | tail -1 | awk '{print $7}' )
      	des_SO2=$( grep "cm^3 (STP)/cm^3 framework" *"$pressure_des".data | head -3 | tail -1 | awk '{print $7}' )
        working_capacity=$(echo "$uptakeSO2 - $des_SO2" | bc)
        
        #Selectivity of SO2:CO2, using uptake...
        
        uptakeCO2=$( grep "cm^3 (STP)/cm^3 framework" *"$pressure_ads".data | head -1 | tail -1 | awk '{print $7}' )
        
	tp=$( awk -v var1=$uptakeSO2 -v var2=$uptakeCO2 'BEGIN { print  ( var1 / var2 ) }' )   #The uptake ratios
	bt=$( awk -v var1=0.002 -v var2=0.198 'BEGIN { print  ( var1 / var2 ) }' )             #The molar ratios
	selectivity=$( awk -v var1=$tp -v var2=$bt 'BEGIN { print  ( var1 / var2 ) }' )        
        
        #SECOND, we need SO2 enthalphy change (kJ/mole)
        enthalpy_ads=$( grep "Enthalpy of adsorption:" *"$pressure_ads".data -A 25 | tail -1 | awk '{print $1}' )
        enthalpy_des=$( grep "Enthalpy of adsorption:" *"$pressure_des".data -A 25 | tail -1 | awk '{print $1}' )
        change_enthalpy=$( echo "$enthalpy_ads - $enthalpy_des" | bc)
        
       
        #AWK CAN DO FLOATING POINT DIVISION WITH e NUMBERS, WE HAVE TO ASSIGN EVEN MORE VARIABLES THOUGH
        top_equation=$( echo "($selectivity - 1) * $working_capacity" | bc)
        bottom_equation=$( echo $change_enthalpy | sed 's/-//g' )
        API=$( awk -v var1=$top_equation -v var2=$bottom_equation 'BEGIN { print  ( var1 / var2 ) }')      
        
	      cd ../../../
        
        echo "$dir,$uptakeSO2,$des_SO2,$uptakeCO2,$working_capacity,$enthalpy_ads,$enthalpy_des,$change_enthalpy,$API" >> API.dat
done

echo "Done: API computed for an adsorption pressure of $pressure_ads Pa and a desorption pressure of $pressure_des" 




