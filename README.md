# Creating a CIFAR 10 Convolution Neural Network

    This repo contains a CNN for training CIFAR 10 Dataset.

`model.py` file contains the CNN Model. It has `BaseNet` model

Model Summary is as Follows -

```
  | Name | Type       | Params
------------------------------------
0 | prep | Sequential | 1.9 K 
1 | c1   | Sequential | 74.0 K
2 | c2   | Sequential | 221 K 
3 | c3   | Sequential | 295 K 
4 | c4   | Sequential | 1.2 M 
5 | c5   | Sequential | 3.5 M 
6 | pool | MaxPool2d  | 0     
7 | fc   | Sequential | 5.1 K 
------------------------------------
5.3 M     Trainable params
0         Non-trainable params
5.3 M     Total params
21.279    Total estimated model params size (MB)
```

Model Accuracy:
- Training Accuracy: 90.4%
- Test Accuracy: 85.7%

Training Accuracy Graph:
![Training Accuracy Graph](<CleanShot 2023-08-04 at 10.28.10@2x.png>)

Training Loss Graph:
![Training Loss Graph](<CleanShot 2023-08-04 at 10.29.04@2x.png>)


Incorrect Predictions
![Incorrect Predictions](<CleanShot 2023-08-04 at 10.30.53@2x.png>)