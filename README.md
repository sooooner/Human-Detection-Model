# Faster R-CNN tensorflow implementation

## Description
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[[link](https://arxiv.org/abs/1506.01497)]

## structure

```
.
└── res/
    ├── data                               
    └── ...                               
└── utils/      
    └── models/      
        ├── __init__.py
        ├── base_model.py                               
        ├── classifier.py                          
        ├── frcnn.py                                 
        ├── layers.py                                 
        ├── rpn.py                                 
        └── train_strp.py       
    ├── __init__.py
    ├── Augmentation.py                           
    ├── intersection_over_union.py                               
    ├── label_generator.py                               
    └── utils.py                                 
├── .gitignore         
├── requirements.txt   
├── config.py                                       # model config
├── description.ipynb                                    # Examples of progress 
├── fast_rcnn.ipynb                                    # Examples of progress 
├── Feature Pyramid Network.ipynb                                    # Examples of progress 
├── README.md                                    # Examples of progress 
└── train.py                                       # model training and save weight py
```

## Usage

```
python GAN.py --model_save=True
```
 
+ --model_save : Whether to save the generated model weight(bool, default=True)  

## Result


## reference
