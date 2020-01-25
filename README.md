# Deep learning for homogenization of two phase high contrast three-dimensional material 
This software is an deep learning application for modeling processing-structure-property (PSP) linkages for two phase high contrast three-dimensional material. It’s a feature-engineering-free framework, which directly takes raw data as input, trains a convolutional neural network (CNN) and outputs output. 

To use this software, what the algorithm requires as input are a numpy array. The shape of this numpy array is (x, 51, 51, 51, 1) where x is the number of microscale volume elements (MVEs) and the dimension of microstructure should be three-dimensional (i.e. 51*51*51). The CNN will establish the PSP linkages in the materials system and predict its macroscale (effective) stiffness.

## Requirements ##
* Python 2.7
* Numpy 1.12.1 (or higher)
* Sklearn 0.18.1 (or higher)
* Keras 2.0.0 (or higher)
* HDF5
* Pickle
* TensorFlow 1.1.0

## Files ##
1. data.pkl: Pickle file. Example data of two-phase high contrast three-dimensional microstructure. It contains 20 MVEs and the dimension of each MVE is 51*51*51.
2. label.pkl: Pickle file. Example data of two-phase high contrast three-dimensional microstructure. It contains the macroscale (effective) stiffness of MVE in data.pkl file. 
3. best_model.h5: HDF5 file. The best CNN model train in this work (see paper in the related publication section). It contains the configuration and weights for the CNN.
4. model.py: Use the best CNN model train in this work (see paper in the related publication section) to directly predict the macroscale (effective) stiffness of the microstructure. 
5. train_model.py: The script to train CNN and its architecture is the same the best CNN train in this work (see paper in the related publication section). To get the best performance on new dataset, users might need to design customized architecture and tune the hyperparameters of CNN.

## How to run it
1. To run model.py: 
	1. Make sure the best_model.h5 in the same folder.
	2. The data file of microstructure should be named as ‘data.pkl’, which is a pickle file. The data should be a numpy array. The shape of bumpy array is (x, 51, 51, 51, 1) where x is the number of microscale volume elements (MVEs) and the dimension of microstructure should be three-dimensional (i.e. 51*51*51). 
	3. To run this file, use commend ‘python model.py’
	4. The predicted results will be save in a Pickle file, named 'predict_result.pkl' in the same folder. 
1. To run train_model.py: 
	1. The data file of microstructure should be named as ‘data.pkl’, which is a pickle file. The data should be a numpy array. The shape of bumpy array is (x, 51, 51, 51, 1) where x is the number of microscale volume elements (MVEs) and the dimension of microstructure should be three-dimensional (i.e. 51*51*51). 
	2. The macroscale (effective) stiffness of microstructure should be named as ‘label.pkl’, which is a pickle file. The data should be a numpy array. The shape of bumpy array is (x, 1) where x is the number of microscale volume elements (MVEs) in the same order as ‘data.pkl’
	3. To run this file, use commend ‘python train_model.py’
	4. The script will save your CNN model in the same folder and the name is ‘my_model.h5’.

## Related Publications ##
1. Z. Yang, Y. C. Yabansu, R. Al-Bahrani, W.-keng Liao, A. N. Choudhary, S. R. Kalidindi, and A. Agrawal, “Deep learning approaches for mining structure-property linkages in high contrast composites from simulation datasets,” Computational Materials Science, vol. 151, pp. 278–287, 2018.

## Contact
* Zijiang Yang (zijiangyang2016@u.northwestern.edu)
* Ankit Agrawal (ankitag@eecs.northwestern.edu)
* Alok Choudhary (choudhar@eecs.northwestern.edu)

## Acknowledgement
This work is supported in part by the following grants: AFOSR award FA9550-12-1-0458; NIST award 70NANB14H012; NSF award CCF-1409601; DOE awards DESC0007456, DE-SC0014330; and Northwestern Data Science Initiative.

