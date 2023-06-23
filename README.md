# SiT: Self-supervised vIsion Transformer 

![](imgs/SiT_.png)


# Requirements
- Create an environment
> conda create -n SiT python=3.8
- Activate the environment and install the necessary packages
> conda activate SiT

> conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

> pip install -r requirements.txt

# Steps to Run the Code
Follow the steps below to successfully run the code in this repository:

1. Download the data by referring to the instructions provided in the [`download_data.ipynb`](./download_data.ipynb) notebook.
2. Convert the dice files into compressed numpy arrays using the [`network_final.mlab`](./network_final.mlab) script in MeVisLab. Make sure to specify the output directory in the 'RunPythonScript' module of the network.
3. Run the [`main_test.py`](./main_test.py) file and make sure to specify the 'data-location' argument.
> python main.py --batch_size 16 --epochs 100 --data-location './data'

**Note:** There are more arguments that can be specified!

# Reference


```
@article{atito2021sit,

  title={SiT: Self-supervised vIsion Transformer},

  author={Atito, Sara and Awais, Muhammad and Kittler, Josef},

  journal={arXiv preprint arXiv:2104.03602},

  year={2021}

}
```


# License
This repository is released under the GNU General Public License.


