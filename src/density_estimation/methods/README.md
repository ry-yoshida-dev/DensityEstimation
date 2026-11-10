# methods

## Overview

Parameter and processor base classes, and the parameters, processors, and enums of each supported density estimation method.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `DensityParameters`, the abstract base class for parameters. |
| [processor.py](./processor.py) | `DensityProcessor`, the abstract base class for processors. |
| [sampling_processor.py](./sampling_processor.py) | `SamplingDensityProcessor`, the abstract base class for processors that can `sample`. |
| [kde/](./kde/README.md) | Kernel density estimation. |
| [mixture/](./mixture/README.md) | Processor and result types shared by the Gaussian mixture methods. |
| [gmm/](./gmm/README.md) | Gaussian mixture fitted by expectation-maximization. |
| [bayesian_gmm/](./bayesian_gmm/README.md) | Gaussian mixture fitted by variational inference. |
| [knn/](./knn/README.md) | k-nearest neighbor density estimation. |
