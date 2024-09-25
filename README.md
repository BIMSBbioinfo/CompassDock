# CompassDock 🧭

<p align="center">
    Navigating Future Drugs with CompassDock 🧭
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2406.06841"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style-for-the-badge&logo=python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style-for-the-badge&logo=PyTorch"/></a>
  <a href="https://pytorch-geometric.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/PyG-2.5.3-3C2179?style-for-the-badge&logo=PyG"/></a>
  <a href="https://lightning.ai/docs/pytorch/stable/starter/installation.html"><img src="https://img.shields.io/badge/Lightning-2.2.4-792EE5?style-for-the-badge&logo=lightning"/></a>
  <a href="https://github.com/facebookresearch/esm"><img src="https://img.shields.io/badge/Meta-ESMFold-0467DF?style-for-the-badge&logo=Meta"/></a>
  <a href=https://github.com/BIMSBbioinfo/Compass/blob/main/LICENSE><img src="https://img.shields.io/badge/License%20-BY--NC--ND--4.0%20-blue"/></a>
</p>

<p float="center">
  <img src="assets/compassdock.gif" width="98%" />
</p>


The [CompassDock](https://arxiv.org/abs/2406.06841) framework is a comprehensive and accurate assessment approach for deep learning-based molecular docking. It evaluates key factors such as the physical and chemical properties, bioactivity favorability of ligands, strain energy, number of protein-ligand steric clashes, binding affinity, and protein-ligand interaction types.


## Quickstart for CompassDock

```bash
conda create --name CompassDock python=3.11 -c conda-forge
conda install -c mx -c conda-forge reduce
conda install -c conda-forge openbabel
conda install -c conda-forge datamol
pip install "fair-esm @ git+https://github.com/asarigun/esm.git",
pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git",
pip install "openfold @ git+https://github.com/asarigun/openfold.git"
pip install compassdock
```

```python
from compassdock import CompassDock
import csv

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


More examples coming soon on Google Colab!


## CompassDock 🧭 in Fine-Tuning Mode <a name="finetuning"></a>

For instructions on how to use Fine-Tuning Mode, please refer to the [previous branch](https://github.com/BIMSBbioinfo/CompassDock?tab=readme-ov-file#compass--in-fine-tuning-mode-)

## Citation

Please cite the following paper if you use this code/repository in your research:
```
@article{sarigun2024compass,
  title={Compass: A Comprehensive Tool for Accurate and Efficient Molecular Docking in Inference and Fine-Tuning},
  author={Sarigun, Ahmet and Franke, Vedran and Akalin, Altuna},
  journal={arXiv preprint arXiv:2406.06841},
  year={2024}
}
```

## Acknowledgements <a name="acknowledgements"></a>
We extend our deepest gratitude to the following teams for open-sourcing their valuable Repos:
* [DiffDock Team](https://github.com/gcorso/DiffDock) (version 2023 & 2024),
* [AA-score Team](https://github.com/Xundrug/AA-Score-Tool),
* [PoseCheck Team](https://github.com/cch1999/posecheck) 

