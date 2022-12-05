# NUCL 575 Final Project - Purdue University

### Files for Final Project:

You can download the files by using the following command:
```bash
git clone https://github.com/marioutiel/NUCL_FP.git
```
It contains 3 python files:

1) **data.py**

To download and prepare data before inputting it to the model

2) **model_train.py**

Where the Neural Network is defined and initialize. Also, it is trained directly with the training set.

Has two parameters:
  - **conv**: (boolean) If we want to train a Convolutional Neural Network (conv=1) or a Fully Connected Neural Network (conv=0)
  - **n_epochs**: (integer) How many epochs we want to use to train the network (default=15)

3) **evaluate.py**

To evaluate the performance of the NN using the test set. Comparing also the train and eval sets

Has one parameter:
  - **conv**: (boolean) If we want to evaluate the CNN (conv=1) or the FCNN (conv=0)

### Data for Final Project

Download from:
https://assets.digitalocean.com/articles/signlanguage_data/sign-language-mnist.tar.gz

Use the following command to download:
```bash
wget https://assets.digitalocean.com/articles/signlanguage_data/sign-language-mnist.tar.gz 
```

### Authors

Made by:
- Alejandro Mayo (amayogar@purdue.edu)
- Elena Gómez
- Lucia Ostolaza (lostolaz@purdue.edu)
- Mario Utiel (mutiel@purdue.edu)



https://notebook.community/marcotcr/lime/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch
