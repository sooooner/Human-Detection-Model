# Human Detection Model
This is a repo that developed a human detection model(Faster-RCNN) using data from the [Motion Keypoint Detection AI Contest](https://dacon.io/competitions/official/235701/overview/description) conducted by dacon.

## Description

Project process description 
+ Description  

## Repo Structure

```
.
└── res/
    ├── data                                # Data doesn't go up to the repo.
    └── ...    
└── model_weight/                        
    ├── weights                             # Weights doesn't go up to the repo.
    └── ...   
└── description/                                 
    └── README.md                           # Project process description
└── utils/      
    └── models/                              
        ├── __init__.py 
        ├── base_model.py                   # Pre-trained backbone      
        ├── classifier.py                   # Faster-RCNN's classifier module  
        ├── frcnn.py                        # Faster-RCNN(Integration of all modules)      
        ├── layers.py                       # Other model layer functions(RoI Pooling, NMS, ...)           
        ├── rpn.py                          # Region Proposal Network module
        └── train_step.py                   # Function for model training in 4 steps
    ├── __init__.py                                
    ├── Augmentation.py                     # Data augmentation(Shift, Flip, Add noise, etc...)  
    ├── intersection_over_union.py          # calculate iou between ground truth and prediction                
    ├── label_generator.py                  # A function that generate the ground truth          
    └── utils.py                            # Other necessary functions
├── .dockerignore                           
├── .gitignore                              
├── Dockerfile                              # A file to create an image for development environment
├── requirements.txt                        # Libraries needed to create images
├── config.py                               # model config py
├── fast_rcnn.ipynb                         # Examples of progress 
├── SQL_loader.ipynb                        # MySQL communication example
├── README.md                               
└── train.py                                # model training and save weight py
```

## Usage
### Data
Download the data through this [link](https://dacon.io/competitions/official/235701/overview/description)

#### Development environment
You don't need to clone this repo, just run Dockerfile and the development environment to run this repo image will be installed.
And just follow the command line described below.

If you use MySQL, you can modify train.py and utils.db_uploader.py a little and use it.
**Refer to SQL_loader.ipynb**

### Docker image build
```terminal
docker build {repo image name}:{tag} .
```
Then, you can create a container, connect it, and use it.

### Training
Running train.py will train the model and store the trained weights.
```terminal
python train.py
```

### inference
When model training is complete, insert your image path and run inference.py, you can see the result.
```terminal
python inference.py --path=./res/{img_nam}
```

## Author
Soon Ho Kwon

github/sooooner
https://tnsgh0101.medium.com/

## License
Copyright © 2016 Jon Schlinkert Released under the MIT license.

