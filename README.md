# SJTU M3DV: Medical 3D Voxel Classification

## EE369 Machine Learning 2019 Autumn Class Competition

This is a kaggle project about the classification of 3D Voxel of lung nodules. I have tried to solve to problem with several networks including: ResNet, VoxNet and DenseNet, as well as several techniques such as mixup etc..

![IMG_0561](picofvoxel/IMG_0561.PNG)

![IMG_0562](picofvoxel/IMG_0562.PNG)



Finally, it is proven by my trials and errors that VoxNet performs the best on this classification problem and I used two of the models trained with different batch sizes and other parameters and combined their results together (one with the weight 0.55 and the other 0.45).

Please use the **ready_to_run.ipynb** file where I wrote a few lines of walk-through codes for users to generate to final result in format .csv.

The models to be used are uploaded to: https://pan.baidu.com/s/15c4_6JaS8CqHQS9irinXvw and the password is: 3i39. The package is about 347MB. Please put the models in the folder **ready_to_run_model/** and then run the walk through code.
