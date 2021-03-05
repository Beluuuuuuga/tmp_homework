# tmp_homework

## Build
This environment is build in Windows(Anaconda) @ Python 3.8.
You make the conda environment and then push below commands:

```
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
$ conda install jupyter
```

## Log

### 1. AEs_Train_001.ipynb  
Train three auto encoder models for stacked auto encoder.  
Three auto encoder models are below.

First model:
```python
def create_AE01_model(k_size):
    input_img = Input(shape=(32, 32, 3))  # 0
    conv1 = Conv2D(64, (k_size, k_size), padding='same', name="Dense_AE01_1")(input_img) # 1
    conv1 = BatchNormalization(name="BN_AE01_1")(conv1) # 2
    conv1 = Activation('relu', name="Relu_AE01_1")(conv1) # 3
        
    decoded = Conv2D(3, (k_size, k_size), padding='same', name="Dense_AE01_2")(conv1) # 4
    decoded = BatchNormalization(name="BN_AE01_2")(decoded) # 5
    decoded = Activation('relu', name="Relu_AE01_2")(decoded) # 6
    return Model(input_img, decoded)
```

Second model:
```python
def create_AE02_model(k_size):
    input_img = Input(shape=(32, 32, 64))  # 0
    conv1 = Conv2D(128, (k_size, k_size), padding='same', name="Dense_AE02_1")(input_img) # 1
    conv1 = BatchNormalization(name="BN_AE02_1")(conv1) # 2
    conv1 = Activation('relu', name="Relu_AE02_1")(conv1) # 3
    pool1 = MaxPooling2D(name="Pool_AE02_1")(conv1)  # 4
    
    unpool1 = UpSampling2D(name="Unpool_AE02_1")(pool1)  # 5
    decoded = Conv2D(64, (k_size, k_size), padding='same', name="Dense_AE02_2")(unpool1) # 6
    decoded = BatchNormalization(name="BN_AE02_2")(decoded) # 7
    decoded = Activation('relu', name="Relu_AE02_2")(decoded) # 8
    return Model(input_img, decoded)
```

Third model:
```python
def create_AE03_model(k_size):
    input_img = Input(shape=(16, 16, 128))  # 0
    conv1 = Conv2D(256, (k_size, k_size), padding='same', name="Dense_AE03_1")(input_img) # 1
    conv1 = BatchNormalization(name="BN_AE03_1")(conv1) # 2
    conv1 = Activation('relu', name="Relu_AE03_1")(conv1) # 3
    pool1 = MaxPooling2D(name="Pool_AE03_1")(conv1)  # 4
    
    unpool1 = UpSampling2D(name="Unpool_AE03_1")(pool1)  # 5
    decoded = Conv2D(128, (k_size, k_size), padding='same', name="Dense_AE03_2")(unpool1) # 6
    decoded = BatchNormalization(name="BN_AE03_2")(decoded) # 7
    decoded = Activation('relu', name="Relu_AE03_2")(decoded) # 8
    return Model(input_img, decoded)
```
- I searched hyperparameters for 1st AE(auto encoder model) by using Optuna.  
These parameters is kernel size (3, 5, 7), optimizer (sgd, adam, rmsprop).  
   

### 2. StackAE_2_CNN_001.ipynb
Train stacked AE to CNN model. 
The stacked AE model and stacked AE encoder to CNN model are below.

Staked AE:
```python
def create_StackedAE01_model(k_size):
    # AE 01 encode
    input_img = Input(shape=(32, 32, 3))  # 0; 32*32*3
    conv1 = Conv2D(64, (k_size, k_size), padding='same', name="Dense_AE01_1")(input_img) # 1; 32*32*64
    conv1 = BatchNormalization(name="BN_AE01_1")(conv1) # 2
    conv1 = Activation('relu', name="Relu_AE01_1")(conv1) # 3

    # AE 02 encode
    conv2 = Conv2D(128, (k_size, k_size), padding='same', name="Dense_AE02_1")(conv1) # 4; 32*32*128
    conv2 = BatchNormalization(name="BN_AE02_1")(conv2) # 5
    conv2 = Activation('relu', name="Relu_AE02_1")(conv2) # 6
    pool1 = MaxPooling2D(name="Pool_AE02_1")(conv2)  # 7; 16*16*128

    # AE 03 encode
    conv3 = Conv2D(256, (k_size, k_size), padding='same', name="Dense_AE03_1")(pool1) # 8; 16*16*256
    conv3 = BatchNormalization(name="BN_AE03_1")(conv3) # 9
    conv3 = Activation('relu', name="Relu_AE03_1")(conv3) # 10
    pool2 = MaxPooling2D(name="Pool_AE03_1")(conv3)  # 11; 8*8*256

    # AE 03 decode
    unpool1 = UpSampling2D(name="Unpool_AE03_1")(pool2)  # 12; 16*16*256
    conv4 = Conv2D(128, (k_size, k_size), padding='same', name="Dense_AE03_2")(unpool1) # 13; 16*16*128
    conv4 = BatchNormalization(name="BN_AE03_2")(conv4) # 14
    conv4 = Activation('relu', name="Relu_AE03_2")(conv4) # 15

    # AE 02 decode 
    unpool2 = UpSampling2D(name="Unpool_AE02_1")(conv4)  # 16; 32*32*128
    conv5 = Conv2D(64, (k_size, k_size), padding='same', name="Dense_AE02_2")(unpool2) # 17; 32*32*64
    conv5 = BatchNormalization(name="BN_AE02_2")(conv5) # 18
    conv5 = Activation('relu', name="Relu_AE02_2")(conv5) # 19

    # AE 01 decode
    decoded = Conv2D(3, (k_size, k_size), padding='same', name="Dense_AE01_2")(conv5) # 20; 32*32*3
    decoded = BatchNormalization(name="BN_AE01_2")(decoded) # 21
    decoded = Activation('relu', name="Relu_AE01_2")(decoded) # 22

    return Model(input_img, decoded)
```

Stacked AE encoder to CNN:
```python
def create_StackedAE01_CNN01_model(encoder):
    input_img = encoder.input
    output = encoder.layers[-1].output # 8,8,256
    x = Conv2D(512,(3,3),padding = "same",activation= "relu")(output)
    x = Conv2D(512,(3,3),padding = "same",activation= "relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = Dense(10,activation = "softmax")(x)

    return Model(input_img, y)
```

- Split train data into train data and validataion data by using staratified.  
Train data and test data is unbalanced, so I cariblate train data and validatation data distribution.  
- Load weights for stacked AE model.  
Weights are created **AEs_Train_001.ipynb**.  
- Train full model (stacked AE encoder to CNN) without data augumentation.  
I train the full model with early stopping.  
`Todo: Print Result and comment`
- Train full model (stacked AE encoder to CNN) with data augumentation.  
I train the full model with early stopping.  
Data augumentation methods are `rotation_range`, `shear_range`, `horizontal_flip`, `vertical_flip`, `width_shift_range`, `height_shift_range`, `zoom_range`, `channel_shift_range`  
`Todo: Print Result and comment`

### 3. AE_2_CNN_002.ipynb
Train single auto encoder to CNN model.
Train stacked AE to CNN model. 
The stacked AE model and stacked AE encoder to CNN model are below.

Single AE:
```python
def create_AE01_model(k_size):
    input_img = Input(shape=(32, 32, 3))  # 0
    conv1 = Conv2D(64, (k_size, k_size), padding='same', name="Dense_AE01_1")(input_img) # 1
    conv1 = BatchNormalization(name="BN_AE01_1")(conv1) # 2
    conv1 = Activation('relu', name="Relu_AE01_1")(conv1) # 3
        
    decoded = Conv2D(3, (k_size, k_size), padding='same', name="Dense_AE01_2")(conv1) # 4
    decoded = BatchNormalization(name="BN_AE01_2")(decoded) # 5
    decoded = Activation('relu', name="Relu_AE01_2")(decoded) # 6
    return Model(input_img, decoded)
```

Singel AE encoder to CNN:
```python
def create_StackedAE01_CNN01_model(encoder):
    input_img = encoder.input
    output = encoder.layers[-1].output # 32,32,64
    x = Conv2D(64,(3,3),padding = "same",activation= "relu")(output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x) # 16,16,64
    
    x = Conv2D(128,(3,3),padding = "same",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128,(3,3),padding = "same",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x) # 8,8,128
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = Dense(10,activation = "softmax")(x)

    return Model(input_img, y))
```

- Split train data into train data and validataion data by using staratified.  
Train data and test data is unbalanced, so I cariblate train data and validatation data distribution.  
- Load weight for single AE model.  
Weight is created **AEs_Train_001.ipynb**.  
- Train full model (single AE encoder to CNN) without data augumentation.  
I train the full model with early stopping.  
`Todo: Print Result and comment`
- Train full model (single AE encoder to CNN) with data augumentation.  
I train the full model with early stopping.  
Data augumentation methods are `rotation_range`, `shear_range`, `horizontal_flip`, `vertical_flip`, `width_shift_range`, `height_shift_range`, `zoom_range`, `channel_shift_range`  
`Todo: Print Result and comment`