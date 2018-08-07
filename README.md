# RAL_GNN
An implementation of "Efficient Relative Attribute Learning using Graph Neural Networks"

## Get started

Download OSR dataset from [http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip](http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip), the subset of Pubfig from [https://www.cc.gatech.edu/~parikh/relative_attributes/relative_attributes_v2.zip](https://www.cc.gatech.edu/~parikh/relative_attributes/relative_attributes_v2.zip). 
Then put the images at './dataset/osr' and './dataset/pubfig'.

Download the pretrained model from [http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put in './pretrain'

## Demo usage
```
python ./src/relative_attributes.py --dataset_name='pubfig' --attr=0
```
