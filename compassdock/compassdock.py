import os
import glob
import pandas as pd
from typing import Optional

from compassdock.inference_wrap import run_docking
from compassdock.compass import run_obabel, run_get_pocket, binding_affinity, posecheck_eval
from compassdock.config import configs


class CompassDock(object):
    def __init__(self,
                args = configs
                ):
        
        r"""
        
        """
        
        self.args = args
        self.results_list = []
        
    
    def recursive_compassdocking(self, 
                protein_path: str,
                protein_sequence: str,
                ligand_description: str,
                complex_name: str, 
                molecule_name: str,
                out_dir: str,
                compass_all: Optional[bool] = False,
                max_redocking_step: Optional[int] = 1,
                save_visualisation: Optional[bool] = False,
                ligand_describe = None, iteration = 0):

        r"""

        """
        
        if iteration >= max_redocking_step:
            print("Maximum iterations reached.")
            return
        
        if protein_sequence is not None and molecule_name is None:
            complex_name = protein_sequence + '_with_' + 'ligand_description'

        elif protein_sequence is not None and molecule_name is molecule_name:
            complex_name = protein_sequence + '_' + molecule_name

        elif protein_path is not None:
            if complex_name is None and molecule_name is None:
                protein_file_name = os.path.splitext(os.path.basename(protein_path))[0]
                complex_name = protein_file_name + '_with_' + 'ligand_description'

            elif complex_name is None and molecule_name is molecule_name:
                protein_file_name = os.path.splitext(os.path.basename(protein_path))[0]
                complex_name = protein_file_name + '_' + molecule_name


        if ligand_describe is None:
            results_summary, protein_path_list, write_dir, best_confidence_score = run_docking(
                self.args,
                protein_path, 
                protein_sequence,
                complex_name,
                ligand_description,
                out_dir,
                save_visualisation
            )
            
        else:

            results_summary, protein_path_list, write_dir, best_confidence_score = run_docking(
                self.args,
                protein_path, 
                protein_sequence,
                complex_name,
                ligand_description,
                out_dir,
                save_visualisation,
                ligand_describe
            )
            
        

        if compass_all:
            sdf_files = glob.glob(os.path.join(write_dir, "*.sdf"))
            sdf_files.sort()

            filtered_sdf_files = []

            for sdf_file in sdf_files:
                if "confidence" in sdf_file:
                    confidence = sdf_file.split("confidence")[-1]
                    confidence = float(confidence.replace(".sdf", ""))
                    if confidence == -1000.00:
                        print(f'Removing file with confidence -1000.00: {sdf_file}')
                        continue
                filtered_sdf_files.append(sdf_file)

            for sdf_file in filtered_sdf_files:
                binding_aff, clashes, strain, confidence_value, inter_dict = self.compassing(  
                    sdf_file, 
                    protein_path_list
                    )
                
                result = {
                    'recursion_step': f'{iteration+1}/{max_redocking_step}',
                    'protein_path': protein_path, 
                    'protein_sequence': protein_sequence,
                    'complex_name': complex_name,
                    'ligand_description': ligand_description,
                    'binding_aff': binding_aff,
                    'clashes': clashes,
                    'strain': strain,
                    'confidence_value': confidence_value,
                    'interactions': inter_dict,
                    'out_dir': f'{sdf_file}'
                }

                self.results_list.append(result)

        else:

            sdf_files = glob.glob(os.path.join(write_dir, 
                                               f"rank1_confidence{best_confidence_score}.sdf"))

            if not sdf_files:
                print(f"No .sdf files found matching the pattern for iteration {iteration}.")
                return

            sdf_file = sdf_files[0] 
            binding_aff, clashes, strain, confidence_value, inter_dict = self.compassing(
                sdf_file, 
                protein_path_list
                )
        
            result = {
                'recursion_step': f'{iteration+1}/{max_redocking_step}',
                'protein_path': protein_path, 
                'protein_sequence': protein_sequence,
                'complex_name': complex_name,
                'ligand_description': ligand_description,
                'binding_aff': binding_aff,
                'clashes': clashes,
                'strain': strain,
                'confidence_value': confidence_value,
                'interactions': inter_dict,
                'out_dir': f'{sdf_file}'
            }

            self.results_list.append(result)

        return self.recursive_compassdocking(protein_path,
                protein_sequence,
                ligand_description,
                complex_name, 
                molecule_name,
                out_dir,
                compass_all,
                max_redocking_step,
                save_visualisation,
                ligand_describe=sdf_file, 
                iteration=iteration + 1)
    
    def results(self):
        r"""

        """
        return self.results_list
    

    def compassing(self, sdf_file, protein_path_list):

        r"""
        
        """

        write_dir = os.path.dirname(sdf_file)
        processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
        pocket_path = os.path.join(write_dir, "pockets/")
        protein_path = protein_path_list[0]

        os.makedirs(processed_sdf_directory, exist_ok=True)
        os.makedirs(pocket_path, exist_ok=True)
        
        file_base_name = os.path.basename(sdf_file).replace('.sdf', '')
        
        if "confidence" in file_base_name:
            try:
                confidence_part = file_base_name.split("confidence")[-1]
                confidence_value = float(confidence_part)
                if confidence_value <= -1000:
                    print(f"Skipping molecule with confidence value {confidence_value}")
                    return 
            except ValueError:
                print("Error extracting confidence value; processing the molecule anyway")
                confidence_value = None
        else:
            confidence_value = None

        protein_name = protein_path.split('/')[-1].replace('.pdb', '')

        input_sdf_path = sdf_file
        output_sdf_path = os.path.join(processed_sdf_directory, f"{file_base_name}_output_clean.sdf")
        
        pdb_name_with_extension = os.path.basename(protein_path)
        pdb_base_name, _ = os.path.splitext(pdb_name_with_extension)

        try:
            clashes, strain, inter_dict = posecheck_eval(protein_path, input_sdf_path)

        except RuntimeError as e:
            if "Element '' not found" in str(e):
                print(f"Error encountered with posecheck_eval for {pdb_base_name}. Assigning default values.")
                clashes = 1000
                strain = 1000
                inter_dict = {'error': 'Element not found, default values assigned'}

            else:
                raise 
        except ValueError as e:  
            print(f"Value error in posecheck_eval for {pdb_base_name}: {e}")
            clashes = 1000
            strain = 1000
            inter_dict = {'error': 'Element not found, default values assigned'}

        try:
            run_obabel(input_sdf_path, output_sdf_path)
        except Exception as e:
            print(f"Obabel error for {pdb_base_name}: {e}")
            score = 1000 

            return score, clashes, strain, confidence_value

        try:
            pocket_path2 = os.path.join(pocket_path, f"{file_base_name}")
            run_get_pocket(protein_path, output_sdf_path, pocket_path2)
        except Exception as e: 
            print(f"Pocket generation failed for {pdb_base_name}: {e}")
            score = 1000 
            return score, clashes, strain, confidence_value


        try:
            pocket_path3 = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
            mol_name, score = binding_affinity(pocket_path3, output_sdf_path)

        except Exception as e: 
            print(f"Error during binding affinity calculation for {pdb_base_name}: {e}")
            score = 1000 

        movin_pckt_pdb = os.path.join(f"{protein_name}_pocket.pdb") 

        try:
            os.remove(movin_pckt_pdb)
        except FileNotFoundError:
            print(f"File not found: {movin_pckt_pdb}")

        return score, clashes, strain, confidence_value, inter_dict
    

    def multiple_run(self,
                    protein_files_dir: str,
                    smiles_file_dir: str,
                    out_dir: str,
                    ligand_description: Optional[str] = None,
                    protein_path: Optional[str] = None,
                    protein_sequence: Optional[str] = None,
                    molecule_name: Optional[str] = None,
                    compass_all: Optional[bool] = False,
                    protein_start: Optional[int] = None,
                    protein_end: Optional[int] = None,
                    smiles_start: Optional[int] = None,
                    smiles_end: Optional[int] = None,
                    max_redocking_step: Optional[int] = 1,
                    save_visualisation: Optional[bool] = False,
                    ):

        if protein_files_dir:
            pdb_files = sorted([file for file in os.listdir(protein_files_dir) if file.endswith('.pdb')])

            if smiles_file_dir:
                smiles_data = pd.read_csv(smiles_file_dir, sep='\t', header=0, on_bad_lines='skip') 
                selected_smiles = smiles_data.loc[smiles_start:smiles_end, 'smiles']
                selected_molecule_name = smiles_data.loc[smiles_start:smiles_end, 'molecule_name']

                for index, smile in selected_smiles.items(): 
                    molecule_name = selected_molecule_name.loc[index]

                    for protein_file in pdb_files[protein_start:protein_end]:
                        protein_path = os.path.join(protein_files_dir, protein_file)
                        ligand_description = smile
                        complex_name = None

                        try:
                            self.recursive_compassdocking(
                                protein_path = protein_path,
                                protein_sequence = protein_sequence,
                                ligand_description = ligand_description,
                                complex_name = complex_name, 
                                molecule_name = molecule_name,
                                out_dir = out_dir,
                                compass_all = compass_all,
                                max_redocking_step = max_redocking_step,
                                save_visualisation = save_visualisation
                                )
                        except Exception as e:
                            print(f"Error processing {protein_file}: {e}")
                            continue

            else:
                for protein_file in pdb_files[protein_start:protein_end]:
                    protein_path = os.path.join(protein_files_dir, protein_file)
                    complex_name = None
                    try:
                        self.recursive_compassdocking(
                            protein_path = protein_path,
                            protein_sequence = protein_sequence,
                            ligand_description = ligand_description,
                            complex_name = complex_name, 
                            molecule_name = molecule_name,
                            out_dir = out_dir,
                            compass_all = compass_all,
                            max_redocking_step = max_redocking_step,
                            save_visualisation = save_visualisation
                            )
                    except Exception as e:
                        print(f"Error processing {protein_file}: {e}")
                        continue

        if smiles_file_dir:
            smiles_data = pd.read_csv(smiles_file_dir, sep='\t', header=0, on_bad_lines='skip') 
            selected_smiles = smiles_data.loc[smiles_start:smiles_end, 'smiles']
            selected_molecule_name = smiles_data.loc[smiles_start:smiles_end, 'molecule_name']

            for index, smile in selected_smiles.items(): 
                molecule_name = selected_molecule_name.loc[index]
                ligand_description = smile
                molecule_name = molecule_name
                complex_name = None

                try:
                    self.recursive_compassdocking(
                        protein_path = protein_path,
                        protein_sequence = protein_sequence,
                        ligand_description = ligand_description,
                        complex_name = complex_name, 
                        molecule_name = molecule_name,
                        out_dir = out_dir,
                        compass_all = compass_all,
                        max_redocking_step = max_redocking_step,
                        save_visualisation = save_visualisation
                        )
                except Exception as e:
                    print(f"Error processing {protein_file}: {e}")
                    continue



