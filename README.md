# Canny Edge Detector
This is a python implementation of the [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector) algorithm.

## Installation
This code runs on [Python](https://www.python.org/downloads/) (version 3.9 or greater) using the [Matplotlib](https://matplotlib.org/stable/users/getting_started/) and [NumPy](https://numpy.org/install/) libraries. This can be done using pip by running the follow commands in the terminal after having installed Python:
```shell
pip install numpy
pip install matplotlib
```

## Running the code
This code can be found at our [GitHub repo](https://github.com/RichartE/cs252-canny-edge-detector.git). After cloning the code locally or downloading it as a .zip. You should run [canny_edge_detector.py](./canny_edge_detector.py) which will launch a gui to select the image you with to run the the edge detector on. Find your image (we have provided a few in the repo) and the algorithm will then run itself. Once done it will display the image at each step in the process, with the lower right one being the final output.

### Adjusting the tunning parameters
You can adjust the tunning parameters by passing the optional flags:

-  -h, --help            show the help message and exit
-  --low-threshold       the gradient value below which pixels are suppressed (float)
- --high-threshold      the gradient value below which pixels are suppressed (float)
- --filter-size         the size of the Gaussian kernel (odd positive integer)
- --filter-sigma        the standard deviation of the Gaussian kernel (float)

### Tuning parameters for provided images


#### Image-Tower
```shell
--low-threshold 205 --high-threshold 225 --filter-size 5 --filter-sigma 0.2
```

#### Image-Tiger
```shell
--low-threshold 20 --high-threshold 100
```

#### Image-Lizard
```shell
--low-threshold 100 --high-threshold 220 
```

#### Image-Anatomy
```shell
--low-threshold 15 --high-threshold 100 
```