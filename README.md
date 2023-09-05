# SiT (Self-supervised vIsion Transformer) for Volumetric Medical Data

This project is an adaptation and training of the SiT model, originally developed by [Sara Atito, Muhammad Awais, and Josef Kittler](https://github.com/Sara-Ahmed/SiT). The original repository provides a foundation for this project, and we have made modifications to suit our specific needs.

Link to the original repository: [Original Repository](https://github.com/Sara-Ahmed/SiT)
</br>Link to the paper: [Paper](https://arxiv.org/abs/2104.03602)

## Objective

In this project, our objective is to adapt and train a self-supervised vision transformer (SiT) for volumetric medical imaging. Leveraging a dataset of 3D CT scans, we aim to harness the power of vision transformers coupled with self-supervised learning (SSL) to learn a meaningful representation of the data. This learned representation can then be utilized to assess the quality of generated medical image data by measuring the distance between synthetic and real volumes in the learned feature space. By doing so, we hope to overcome the limitations of traditional evaluation metrics and provide a more reliable and domain-specific assessment of the generated medical images.

### Data 
For this study, we used data from the Lung Image Database Consortium (LIDC) Collection, hosted within the IDC repository. The LIDC database contains spiral CT lung scans with marked-up annotations of lesions, specifically designed to aid research on lung cancer detection and diagnosis.
The Imaging Data Commons (IDC) serves as a data repository and platform for sharing cancer imaging data, created as part of the Cancer Research Data Commons (CRDC) initiative by the National Cancer Institute (NCI) in the United States.

Data Source: [NCI Imaging Data Commons](https://aacrjournals.org/cancerres/article/81/16/4188/670283/NCI-Imaging-Data-CommonsNCI-Imaging-Data-Commons)



![](imgs/architecture_new.png)


# Requirements
- Create an environment
> conda create -n SiT python=3.8
- Activate the environment and install the necessary packages
> conda activate SiT

> conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

> pip install -r requirements.txt

# Steps to Run the Code

## Prepare the Data
Follow the steps below to successfully run the code in this repository:

- Download the data by referring to the instructions provided in the [`download_data.ipynb`](./download_data.ipynb) notebook.

    **Note:** To clean the file structure follow the steps in the next section (Clean File Structure).

- Convert the DICOM files into compressed NumPy arrays. There are two possible methods you can use:
  - **MeVisLab:** Using the [`network_final.mlab`](./network_final.mlab) in MeVisLab. Make sure to specify the output directory in the `RunPythonScript` module of the network.
  - **Python:** Run the [dicom_to_npz.py](./dicom_to_npz.py) file and make sure to specify the `clean_folder_dir`(directory containing the DICOM files in a clean structure.), and `output_dir` (directory where the output files will be saved). Optionally, you can specify the voxel size. The default value is 2,2,2.
    
    > python dicom_to_npz.py --clean_folder_dir /path/to/sorted_folder --output_dir /path/to/output_folder --voxel_size 2,2,2

## Prepare and conduct the actual training

- Obtain and set `Weights and Biases` API key
  - Create an account at wandb.ai
  - You'll find your API key in the start page (a kind of default readme)
  - Log into W&B through the command line (using wandb login --relogin), and provide the API key.
- Run the [`main_3D.py`](./main_test.py) file and make sure to specify the 'data_location' argument.
  > python main_test.py --batch_size 16 --epochs 100 --data_location './data'

  **Note:** There are more arguments that can be specified!


# Clean File Structure
Follow the steps below to organize the flat list of DICOM files into the PatientID-StudyInstanceUID-SeriesInstanceUID-SOPInstanceUID hierarchy.

>git clone https://github.com/pieper/dicomsort.git

>pip install pydicom

>python dicomsort/dicomsort.py -u dicom_files_dir cohort_sorted/%PatientID/%StudyInstanceUID/%SeriesInstanceUID/%SOPInstanceUID.dcm


**Note:** Replace `dicom_files_dir` with the directory of saved dicom files.

# FAQ and Troubleshooting

* If you use WSL2 on a Windows system, you might encounter an error `Libcudnn_cnn_infer.so.8 library can not be found`. The most likely reason is that the `LD_LIBRARY_PATH` isn't set up sufficiently in WSL2. Run the following command, or add it to .bashrc: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`. (If this does not help, check if the path is correct using `ldconfig -p | grep cuda`). [Reference](https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661).
* If you interrupt the training and your training process keeps blocking the distributed training socket, you may need to kill the process manually. On the command line, use `kill $ (lsof -t -i:<PortNumber>)` (Port is 29500 for this code.)

# Reference


```
@article{atito2021sit,

  title={SiT: Self-supervised vIsion Transformer},

  author={Atito, Sara and Awais, Muhammad and Kittler, Josef},

  journal={arXiv preprint arXiv:2104.03602},

  year={2021}

}
```


