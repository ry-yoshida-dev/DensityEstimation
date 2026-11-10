# mixture

## Overview

Processor and result types shared by `gmm/` and `bayesian_gmm/`.

## Components

| Component | Description |
|-----------|-------------|
| [sampling_processor.py](./sampling_processor.py) | `MixtureProcessor`, which adds fitted components, information criteria, and component probabilities. |
| [components.py](./components.py) | `MixtureComponents`: fitted weights, means, and covariances. |
| [information_criteria.py](./information_criteria.py) | `InformationCriteria`: AIC and BIC. |
| [covariance_type.py](./covariance_type.py) | `MixtureCovarianceType` enum. |
