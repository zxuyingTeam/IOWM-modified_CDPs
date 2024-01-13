# Application of IOWM Method on Low-does CT

Code for IOWM method on **Low-does CT**

## Requirements:

- Linux: Ubuntu 18.04
- cuda9.2 & cudnn7.5.0
- Python 3.7.4
- torch 1.3.1 (pytorch)
- torchvision 0.4.0
- numpy 1.16.2
- matplotlib
- scipy 1.3.1
- glob
- pydicom

## How to run the code

The root directory of this project is "Low_Dose", and the data needs to be processed according to the following path

```
Low_Dose
├── data
│   ├── Mayo_abdomen
│   │   ├── L**
│   │   │   ├── L**input.raw
│   │   │   ├── ...
│   │   │   └── L**target.raw
│   │   ├── ...
│   │   ├── npy_img_Poisson_noise_1e4
│   │   │   └── L**
│   │   │       ├── L**input.raw
│   │   │       └── L**target.raw
│   │   └── npy_img_Poisson_noise_1e5
│   │       └── L**
│   │           ├── L**input.raw
│   │           └── L**target.raw
│   ├── Mayo_chest
│   │   └── L**
│   │       ├── input
│   │       │   └── L**input.raw
│   │       └── target
│   │           └── L**target.raw
│   └── npy_img
│       ├── L**_pic-id_input.raw
│       └── L**_pic-id_target.raw
└── IOWM
    ├── Poisson_noise_1e4
    │   ├── nosie_1
    │   │   └── loss__.npy
    │   └── nosie_2
    │       └── loss__.npy
    └── fig
        ├── old_**.raw
        ├── label_**.raw
        └── pred_**.raw
```

The `data` folder stores the original data and data converted to numpy format, while the `IOWM` folder saves model parameters, loss values during experiments, and some images from the experimental period. In this project, files with the prefix `prep` can convert chest and abdominal CT raw data into numpy format. Running files with the prefix `main` can execute the network training and testing parts of the project.



