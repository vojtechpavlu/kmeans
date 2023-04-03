# K-Means


K-Means is a popular cluster algorithm widely used in various fields for data
analysis. Its centroid-based nature enables the simple way of assignment of
each given instance into any of the found clusters.

To find more information about this algorithm, have a look at the 
[Wikipedia page](https://en.wikipedia.org/wiki/K-means_clustering).

This implementation is not meant to be used in a production environments,
because it uses just the simple Python means; no usage of NumPy or any other
scientific and compute performance optimized library. The purpose of this 
project is to be rather educational and not for being used in any mission 
critical applications.


## Requirements

The only dependency required by this project is a `pytest` library for testing
purposes. If you are not about to run these tests, you can skip it. Otherwise,
clone this repository to your device (into a virtual environment, if you want) 
and run:

```bash
  python -m pip install --upgrade pip
  python -m pip install pytest
```

On some devices, this might be kinda problematic - for example on linux OS.
So if the previous does not work, try these commands:

```bash
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest
```


## Project structure

This project is organized into simple multi-directory structure as seen
below:

- `.github` - contains just a simple workflow to run all the tests using
GitHub Actions
  
- `src` - contains the actual source code, the implementation

- `test` - contains the unit tests to validate some of the features  


## Notable objects

The most important classes are definitely 

- `Point` and `Centroid` (both defined in the `./src/datapoint.py` module), 
  
- abstract class `Metric` used to calculate the distance between two points 
  in a multidimensional space (`./src/datapoint.py`) 
  
- class `KMeans`, that defines instances providing the actual model 
  (this is defined in the `./src/k_means.py` module).