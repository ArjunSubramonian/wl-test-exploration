# Weisfeiler and Lehman Go Measurement Modeling: Probing the Validity of the WL Test

This repository is the official implementation of `Weisfeiler and Lehman Go Measurement Modeling: Probing the Validity of the WL Test`. 

## Requirements

To install requirements, use the provided `requirements.txt`:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To run the experiments on the alignment of 1-WL colorings with GIN representations, simply run all the cells in `representations-exploration.ipynb`. To run all other experiments in the "Benchmarking expressive power" section of the paper, run all the cells in `graph-benchmark-exploration.ipynb`. To generate the survey plots, run the cells in `survey-analysis.ipynb`.

## License

Some utilities code files are adapted from the [PyTorch Geometric library](https://github.com/pyg-team/pytorch_geometric) and [nifty repository](https://github.com/chirag126/nifty); these files contain the appropriate copyright notices. All other code is written by the authors.
