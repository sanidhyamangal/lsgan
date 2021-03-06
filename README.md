# LSGAN
This repository is a simple implementation of Least Squares GAN (LSGAN) trained on fashion mnist dataset

More information about it could be found on [link](https://arxiv.org/abs/1611.04076)

![alt LSGAN](./lsgan.png "LSGAN")

### requirements
- tensorflow >= 2.0
- pillow
- matplotlib

### Preparing virtual env
If you dont have venv installed in your system you can do it by using
```
sudo apt install python3-venv
```
Now create virtual enviornment for your application
```
python -m venv env
```

Activating the venv
- Ubuntu/Linux
```
source env/bin/activate
```

- Windows 
```
.\env\Scripts\activate
```

### Downloading the dataset
Dataset can be downloaded from following [link](https://gitlab.com/sanidhyamangal/datasets/-/raw/master/fashion-mnist_train.csv)

In case you are an Linux\Unix user you can also use following command for downloading the dataset
```
wget https://gitlab.com/sanidhyamangal/datasets/-/raw/master/fashion-mnist_train.csv
```

### How to train model
```
python3 main.py
```

### How to Generate Images?
Image generation can be carried out using two different methods, which are as follows:

#### Using shell
```bash
python -i generate_images.py

image_generator = LSGANImageGenerator()
image_generator.generate_save_images()
```

#### In a python file
```python
from generate_images import LSGANImageGenerator

image_generator = LSGANImageGenerator()
image_generator.generate_save_images()
```