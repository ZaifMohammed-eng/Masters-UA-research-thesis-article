import numpy as np
import pandas as pd
from tc_python import TCPython, ThermodynamicQuantity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def list_stable_phases_and_calculations(results, temperature, elements):
    """Extract phase data including mole fractions for each element."""
    data = []
    stable_phases = results.get_stable_phases()
    phase_data = {}

    logging.info(f"Stable phases found: {stable_phases}")

    for phase in stable_phases:
        try:
            amount = results.get_value_of(f'NP({phase})')
            molar_volume = results.get_value_of(f'VM({phase})') * 1e6  # Convert from m^3 to cm^3

            # Get mole fractions for each element in this phase
            mole_fractions = {}
            for component in results.get_components():
                try:
                    # X(phase_name, element) returns mole fraction of element in phase
                    mole_fraction = results.get_value_of(f'X({phase},{component})')
                    mole_fractions[component] = mole_fraction
                    logging.info(f"Phase {phase}, component {component}: mole_fraction={mole_fraction}")
                except Exception as e:
                    logging.warning(f"Could not get mole fraction for {component} in {phase}: {e}")
                    mole_fractions[component] = np.nan

            data.append([phase, amount, molar_volume])
            phase_data[phase] = {
                'amount': amount,
                'molar_volume': molar_volume,
                'mole_fractions': mole_fractions
            }
            logging.info(
                f"Phase {phase}: amount={amount}, molar_volume={molar_volume}, mole_fractions={mole_fractions}")
        except Exception as e:
            logging.error(f"Error processing phase {phase}: {e}")

    return data, phase_data


input_file = 'pareto_solutions_mobo'

# Read the CSV file
try:
    systems_df = pd.read_csv(input_file)
    logging.info(f"Successfully read {len(systems_df)} systems from {input_file}")
    logging.info(f"Columns: {list(systems_df.columns)}")
except Exception as e:
    logging.error(f"Error reading input file: {e}")
    raise

output_results = []

from tc_python.single_equilibrium import SingleEquilibriumOptions, SingleEquilibriumCalculation

with TCPython() as start:
    options = SingleEquilibriumOptions()
    options.set_required_accuracy(1e-6)
    options.set_smallest_fraction(1e-12)
    options.set_max_no_of_iterations(500)
    options.set_global_minimization_max_grid_points(2000)
    options.enable_control_step_size_during_minimization()
    options.enable_force_positive_definite_phase_hessian()
    options.enable_approximate_driving_force_for_metastable_phases()

    # Iterate through each row
    for index, system in systems_df.iterrows():
        try:
            temperature = system['Temperature(K)']

            # Get mass fractions, excluding non-element columns
            mass_fractions = {}
            for element in systems_df.columns:
                if element not in ['Temperature'] and not pd.isna(system[element]):
                    mass_fractions[element] = system[element]

            logging.info(
                f"Processing system {index + 1} with elements: {list(mass_fractions.keys())} and temperature: {temperature}")

            if not mass_fractions:
                logging.warning(f"System {index + 1}: No valid mass fractions found, skipping.")
                continue

            element_to_ignore = max(mass_fractions, key=mass_fractions.get)
            logging.info(f"System {index + 1}: Ignoring element {element_to_ignore} for mass fraction condition.")

            calculation = (
                start
                .select_database_and_elements("TCHEA7", list(mass_fractions.keys()))
                .get_system()
                .with_single_equilibrium_calculation()
                .set_condition(ThermodynamicQuantity.temperature(), temperature)
                .set_condition(ThermodynamicQuantity.pressure(), 101325)
            )

            for element, mass_fraction in mass_fractions.items():
                if element != element_to_ignore:
                    calculation.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component(element),
                                              mass_fraction)

            calculation.enable_global_minimization()
            calculation.with_options(options)

            logging.info(f"System {index + 1}: Starting calculation")
            results = calculation.calculate()
            logging.info(f"System {index + 1}: Calculation completed")

            # Pass elements list to the function so it can extract mole fractions
            data, phase_data = list_stable_phases_and_calculations(results, temperature, list(mass_fractions.keys()))

            # Create result dictionary starting with all input columns
            result = {}

            # Add all columns from the input CSV row
            for column in systems_df.columns:
                result[column] = system[column]

            # Add each phase amount and molar volume as separate columns
            if phase_data:
                logging.info(f"System {index + 1}: phase_data keys = {list(phase_data.keys())}")

                for phase, phase_info in phase_data.items():
                    result[f"{phase}_amount"] = phase_info['amount']
                    result[f"{phase}_molar_volume"] = phase_info['molar_volume']

                    # Add Al and Cr mole fractions for each phase with full phase name
                    if phase.startswith("FCC_L12#") or phase.startswith("BCC_B2#"):
                        for element in ['AL', 'CR']:
                            mole_frac = phase_info.get('mole_fractions', {}).get(element, np.nan)
                            result[f"{phase}_Mole_Fraction_{element}"] = mole_frac
                            logging.info(f"System {index + 1}: {phase} {element} mole fraction = {mole_frac}")
            else:
                logging.warning(f"System {index + 1}: No phase data found")

            logging.info(f"System {index + 1}: Result keys = {list(result.keys())}")
            output_results.append(result)

        except Exception as e:
            logging.error(f"System {index + 1}: Error during processing: {e}", exc_info=True)
            continue

# Create DataFrame and save
if output_results:
    output_df = pd.DataFrame(output_results)

    try:
        output_df.to_csv("Pareto_Front.csv", index=False)
        logging.info("Results successfully saved to 'Simulated_optimized_alloys.csv'.")
        logging.info(f"Total systems processed: {len(output_results)}")
        logging.info(f"Output columns: {list(output_df.columns)}")
    except Exception as e:
        logging.error(f"Error saving output to CSV: {e}")
else:
    logging.warning("No results to save.")