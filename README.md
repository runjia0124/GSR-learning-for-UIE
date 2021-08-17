# Global Structure Guided Recovery Model

This is the official implementation for "Global Structure-guided Learning Framework for Underwater Image Enhancement".

### Representative Results

![](https://raw.githubusercontent.com/runjia0124/GSR-learning/main/archive/display.png)

### Overall Architecture

![](https://raw.githubusercontent.com/runjia0124/GSR-learning/main/archive/pipeline.png)

## Environment Preparing

- python == 3.7
- torchvision == 0.7.0
- pytorch == 1.6.0
- cudatoolkit == 10.2

### Training process
Please prepare your dataset and put the input image into `./dataset/trainA`, the reference image into `./dataset/trainB`.

`python scripts/script.py --train`

### Test

` python scripts/script.py --test`

### Reach me

E-mail: junko.lin@yahoo.com
