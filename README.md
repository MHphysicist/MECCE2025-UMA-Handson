# Practical Introduction to AI in Materials Discovery

This repository contains hands-on materials for working with the UMA (Unified Machine learning potential for Atomistic simulations) model, part of the "Practical Introduction to AI in Materials Discovery" and "Hands-on Practical Training: Meta’s Universal Models for Atoms (UMA) Models for Corrosion Management" workshops at The 19th Middle East Corrosion Conference and Exhibition (MECC 2025). The conference is held from November 11-13, 2025, at Dhahran Expo, Kingdom of Saudi Arabia.

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/MHphysicist/MECC2025-UMA-Handson.git
   cd MECC2025-UMA-Handson
   ```
   
   Alternatively, you can [download the ZIP file](https://github.com/MHphysicist/MECCE2025-UMA-Handson/archive/refs/heads/main.zip) and extract it.

## Overview

These materials focus on practical applications of UMA for materials discovery and catalysis simulations. Through hands-on exercises, you'll learn to:
- Set up and use pretrained AI models for materials prediction
- Optimize catalyst performance using machine learning
- Predict material properties and behaviors
- Screen potential materials for applications like CO₂ capture

The exercises are designed to provide immediate practical experience with AI-driven methodologies that you can apply to your own materials research and development projects.

## Author

**Muhammad H. M. Ahmed**  
Department of Materials Science & Engineering  
King Fahd University of Petroleum and Minerals (KFUPM)

**Contact:**  
- Email: husseinphysicist@gmail.com | g202318650@kfupm.edu.sa
- Phone: +966 53 358 4744
- LinkedIn: [Muhammad H. M. Ahmed](https://www.linkedin.com/in/muhammad-h-m-ahmed/)

## Prerequisites

Before starting the workshop, you'll need to:
1. Install Conda (for managing Python environments)
2. Create a Hugging Face account and get access to UMA
3. Set up your environment with the required packages
4. You can find a full walkthrough in this [link](https://www.youtube.com/watch?v=SQl3jxb6HRw)

## Installing Conda

### Windows
1. Download Miniconda3:
   - Visit [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)
   - Download the Windows 64-bit installer for Python 3.11
   - Direct link: [Miniconda3 Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

2. Install Miniconda3:
   - Run the downloaded .exe file
   - Select "Install for all users" (recommended)
   - Use default installation path
   - Check "Add Miniconda3 to my PATH environment variable"
   - Check "Register Miniconda3 as my default Python"

3. Verify installation:
   - Open a new Anaconda Prompt
   - Run: `conda --version`

### macOS
1. Download Miniconda3:
   - Open Terminal
   - Download installer:
     ```bash
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # For Apple Silicon
     # OR
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # For Intel Mac
     ```

2. Install Miniconda3:
   ```bash
   bash Miniconda3-latest-MacOSX-*.sh
   ```
   - Press ENTER to review license
   - Type "yes" to accept
   - Confirm installation location
   - Type "yes" to initialize Miniconda3

3. Verify installation:
   - Open a new terminal window
   - Run: `conda --version`

### Linux
1. Download Miniconda3:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. Install Miniconda3:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   - Press ENTER to review license
   - Type "yes" to accept
   - Confirm installation location
   - Type "yes" to initialize Miniconda3

3. Verify installation:
   - Open a new terminal
   - Run: `conda --version`

## Environment Setup

1. Create a new conda environment with Python 3.11.9:
```bash
conda create -n mecc-uma python=3.11.9 -y
```

2. Activate the environment:
```bash
conda activate mecc-uma
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. NGLView Setup:
   NGLView is pre-configured in JupyterLab 4 and should work out of the box. If you experience any visualization issues:
   ```bash
   pip uninstall nglview -y
   pip install nglview==3.1.4
   ```
   Then restart JupyterLab.

5. Verify installation:
```bash
   # Should show Python 3.11.9
   python --version

   # Should show all required packages
   conda list
```

## Hugging Face Setup

### 1. Create Account
```markdown
1. Visit [huggingface](https://huggingface.co/join)
2. Fill in your details:
   - Username
   - Email
   - Password
3. Click "Create account"
4. Verify your email address

Note: If you already have an account, sign in at [huggingface login](https://huggingface.co/login)
```

### 2. Request Access to UMA Model
```markdown
1. Visit [facebook/UMA model page](https://huggingface.co/facebook/UMA)
2. Click "Access Request" button
3. Fill in the request form:
   - Select "Academic/Research" for use case
   - Briefly describe your use: "MECC 2025 Workshop: Practical Introduction to AI in Materials Discovery"
   - Accept the terms of use
4. Submit and wait for approval (approval time varies)
```

### 3. Create Access Token
```markdown
1. Go to [Access Tokens page](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Fill in token details:
   - Name: "uma-workshop" (or any memorable name)
   - Role: Select "read" scope (this is sufficient for the workshop)
4. Click "Generate token"
5. COPY YOUR TOKEN IMMEDIATELY - you won't be able to see it again!
```

### 4. Set Up Token Locally

The Hugging Face CLI provides a secure and consistent way to manage your token across all platforms:

1. Open a terminal (make sure you're in the `mecc-uma` conda environment):
   ```bash
   conda activate mecc-uma
   ```

2. Log in using the CLI:
   ```bash
   huggingface-cli login
   ```

3. When prompted, paste your token and press Enter

This will securely store your token in:
- Windows: `%USERPROFILE%\.huggingface\token`
- macOS/Linux: `~/.huggingface/token`

### 5. Verify Setup

Check if you're logged in:
```bash
huggingface-cli whoami
```

You should see your Hugging Face username displayed.

**Important Security Notes:**
- Never share your token or token file
- If token is compromised, revoke it immediately on Hugging Face website
- To logout and remove token:
  ```bash
  huggingface-cli logout
  ```

## Verifying the Setup

1. Navigate to the test directory:
   ```bash
   cd "0-Test_UMA"
   ```

2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Open `Test_UMA_Setup.ipynb` and run all cells to verify:
   - Package imports work correctly
   - UMA model loads successfully
   - Simple H2 optimization runs without errors

If all cells run successfully, your environment is ready for the workshop!

## Directory Structure

```
.
├── requirements.txt           # Full package dependencies
├── README.md                 # This file
├── 0-Test_UMA/              # Initial setup verification
│   └── Test_UMA_Setup.ipynb # Test notebook
└── 1-Catalysis_Hands_on/    # Main workshop materials
    ├── NH3_Decomposition_MECCE_Workshop.ipynb  # Main tutorial notebook
    ├── utility.py           # Helper functions
    └── Isolated_Molecules/  # Reference molecule structures
        ├── H2/
        ├── NH2/
        └── NH3/
```

## Common Issues and Solutions

### Conda and Environment Issues
1. **Conda not recognized**
   - Close and reopen your terminal after installation
   - For Windows: Verify "Add to PATH" was checked during installation
   - For macOS/Linux: Run `source ~/.bashrc` or `source ~/.zshrc`
   - Verify with: `conda --version`

2. **Environment Issues**
   - Verify environment is active (should see `(mecc-uma)` in prompt)
   - List environments: `conda env list`
   - If activation fails: Run `conda init` and restart terminal
   - To reactivate: `conda activate mecc-uma`

3. **Package Installation Issues**
   ```bash
   # First, ensure you're in the correct environment
   conda activate mecce-uma
   
   # If fairchem fails
   pip install --force-reinstall fairchem-core
   
   # For dependency issues
   conda clean --all
   pip install -r requirements.txt
   ```

4. **NGLView Issues**
   ```bash
   # Verify installation and version
   conda list nglview
   
   # If visualization issues occur
   pip uninstall nglview -y
   pip install nglview==3.1.4
   
   # Restart JupyterLab
   ```

5. **Hugging Face Token Issues**
   ```bash
   # Verify login status
   huggingface-cli whoami
   
   # If not logged in or token expired:
   huggingface-cli login
   
   # To remove and reset token:
   huggingface-cli logout
   huggingface-cli login
   ```

6. **CUDA/GPU Issues**
   - Workshop examples use CPU by default
   - For GPU support:
     1. Install CUDA toolkit
     2. Install GPU version of PyTorch
     3. Update device in notebooks from "cpu" to "cuda"

For additional help, contact the workshop organizers or consult the FairChem documentation at [fair-chem.github.io](https://fair-chem.github.io).

## License

This workshop material is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means you are free to:
- Share and adapt the material for non-commercial purposes
- Must give appropriate credit and indicate if changes were made

Commercial use is not permitted. See the [LICENSE](LICENSE) file for details.

Note: The UMA model has its own license terms which you accept during the access request process.

## Citation

If you use these workshop materials in your research, please cite:

The workshop materials:
```bibtex
@inproceedings{ahmed2025materials,
    author = {Ahmed, Muhammad H. M.},
    title = {UMA Hands-on Materials: Practical Introduction to AI in Materials Discovery},
    booktitle = {The 19th Middle East Corrosion Conference and Exhibition},
    series = {MECC},
    year = {2025},
    month = {November},
    address = {Dhahran Expo, Dhahran, Saudi Arabia},
    organization = {Materials Engineering Association and AMPP Dhahran Saudi Arabia Chapter},
    url = {https://github.com/MHphysicist/MECCE2025-UMA-Handson}
}
```

The UMA model:
```bibtex
@article{wang2023uma,
    title={UMA: A Unified AI Model for Catalysis},
    author={Wang, Yufeng and Guo, Shule and Felton, Katie and Kang, Jiangyan and Fung, Victor and others},
    journal={arXiv preprint arXiv:2308.04608},
    year={2023}
}
```

## Acknowledgments

- Dr. Abduljabar Q. Al-Sayoud, research advisor at KFUPM, for his academic supervision and valuable guidance
- Meta Research for developing and open-sourcing the UMA model
- King Fahd University of Petroleum and Minerals (KFUPM)
- The FairChem team for their excellent software infrastructure
- MECC Conference organizers and participants

## Support & Help

### Workshop Materials & Technical Issues
- **Direct Support**: 
  - Muhammad H. M. Ahmed
  - Email: husseinphysicist@gmail.com | g202318650@kfupm.edu.sa
  - Phone: +966 53 358 4744
  - LinkedIn: [Muhammad H. M. Ahmed](https://www.linkedin.com/in/muhammad-h-m-ahmed/)

### Additional Resources
- UMA Model: [facebook/UMA on Hugging Face](https://huggingface.co/facebook/UMA)
- FairChem Documentation: [fair-chem.github.io](https://fair-chem.github.io)
- ASE Documentation: [wiki.fysik.dtu.dk/ase](https://wiki.fysik.dtu.dk/ase/)

### Common Issues
For common setup and runtime issues, please check the [Common Issues and Solutions](#common-issues-and-solutions) section above.

## References

- [FairChem Documentation](https://fair-chem.github.io)
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [UMA Model Paper](https://arxiv.org/abs/2308.04608)
- [Miniconda Documentation](https://docs.conda.io/en/latest/miniconda.html)
- [Hugging Face Documentation](https://huggingface.co/docs)
