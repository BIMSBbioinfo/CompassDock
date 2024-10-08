# CompassDock 🧭

<p align="center">
    Navigating Future Drugs with CompassDock 🧭
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2406.06841"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a>
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style-for-the-badge&logo=python"/></a>
  <a href="https://pypi.org/project/compassdock/"><img src="https://img.shields.io/badge/PyPI%20-package%20-3775A9?style-for-the-badge&logo=PyPI"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style-for-the-badge&logo=PyTorch"/></a>
  <a href="https://pytorch-geometric.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/PyG-2.5.3-3C2179?style-for-the-badge&logo=PyG"/></a>
  <a href="https://github.com/facebookresearch/esm"><img src="https://img.shields.io/badge/Meta-ESMFold-0467DF?style-for-the-badge&logo=Meta"/></a>
  <a href="https://github.com/BIMSBbioinfo/CompassDock/blob/cd_main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-EF9421?style-for-the-badge&logo=Creative-Commons"/></a>
  <a href="https://colab.research.google.com/drive/1h-ArOH6EsG-d-ZG5SQDQJxcZPyJwiTVb?usp=sharing"><img src="https://img.shields.io/badge/Google_Colab-Tutorial-F9AB00?style-for-the-badge&logo=googlecolab"/></a>
</p>

<p float="center">
  <img src="https://raw.githubusercontent.com/BIMSBbioinfo/CompassDock/refs/heads/cd_main/assets/compassdock.png" width="98%" />
</p>


The [CompassDock](https://arxiv.org/abs/2406.06841) framework is a comprehensive and accurate assessment approach for deep learning-based molecular docking. It evaluates key factors such as the physical and chemical properties, bioactivity favorability of ligands, strain energy, number of protein-ligand steric clashes, binding affinity, and protein-ligand interaction types.


## Quickstart for CompassDock

```bash
conda create --name CompassDock python=3.11 -c conda-forge
conda install -c ostrokach-forge reduce
conda install -c conda-forge openbabel
conda install -c conda-forge datamol
pip install compassdock
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install "fair-esm @ git+https://github.com/asarigun/esm.git"
pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
pip install "openfold @ git+https://github.com/asarigun/openfold.git"
```

```python
from compassdock import CompassDock

cd = CompassDock()

cd.recursive_compassdocking(
        protein_path = 'example/proteins/1a46_protein_processed.pdb',
        protein_sequence = None,
        ligand_description = 'CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1',
        complex_name = 'complex_1',
        molecule_name = 'molecule_1',
        out_dir = 'results',
        compass_all = False)

results = cd.results()

print(results)
```

### Protein Sequance - Ligand Docking 

```python
from compassdock import CompassDock

# Initialize CompassDock
cd = CompassDock()

# Perform docking using the provided protein and ligand information
cd.recursive_compassdocking(
        protein_path = None,
        protein_sequence = 'GIQSYCTPPYSVLQDPPQPVV',
        ligand_description = 'CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1',
        complex_name = 'complex_1',
        molecule_name = 'molecule_1',
        out_dir = 'results',
        compass_all = False,
        max_redocking_step = 1)

# Retrieve and print the docking results
results = cd.results()

print(results)
```


More examples can be found in <a href="https://colab.research.google.com/drive/1h-ArOH6EsG-d-ZG5SQDQJxcZPyJwiTVb?usp=sharing"><img src="https://img.shields.io/badge/Google_Colab-Tutorial-F9AB00?style-for-the-badge&logo=googlecolab"/></a>!


## CompassDock 🧭 in Fine-Tuning Mode <a name="finetuning"></a>

For instructions on how to use Fine-Tuning Mode, please refer to the [previous branch](https://github.com/BIMSBbioinfo/CompassDock/tree/master?tab=readme-ov-file#datasets--)

## Citation

Please cite the following paper if you use this code/repository in your research:
```
@article{sarigun2024compass,
  title={CompassDock: Comprehensive Accurate Assessment Approach for Deep Learning-Based Molecular Docking in Inference and Fine-Tuning},
  author={Sarigun, Ahmet and Franke, Vedran and Uyar, Bora and Akalin, Altuna},
  journal={arXiv preprint arXiv:2406.06841},
  year={2024}
}
```

## Acknowledgements <a name="acknowledgements"></a>
We extend our deepest gratitude to the following teams for open-sourcing their valuable Repos:
* [DiffDock Team](https://github.com/gcorso/DiffDock) (version 2023 & 2024),
* [AA-score Team](https://github.com/Xundrug/AA-Score-Tool),
* [PoseCheck Team](https://github.com/cch1999/posecheck) 

