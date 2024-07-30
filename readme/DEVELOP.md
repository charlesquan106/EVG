# Develop

This document provides tutorials to develop CenterNet. `lib/src/opts` lists a few more options that the current version supports.

## New dataset
Basically there are three steps:
- Create a dataset intilization file in `src/lib/datasets/dataset`. In most cases you can just copy `src/lib/datasets/dataset/eve.py` to your dataset name and change the category information, and annotation path.
- Import your dataset at `src/lib/datasets/dataset_factory`.

## New task

You will need to add files to `src/lib/datasets/sample/`, `src/lib/datasets/trains/`, and `src/lib/datasets/detectors/`, which specify the data generation during training, the training targets, and the testing, respectively.

## New architecture

- Add your model file to `src/lib/models/networks/`. The model should accept a dict `heads` of `{name: channels}`, which specify the name of each network output and its number of channels. Make sure your model returns a list (for multiple stages. Single stage model should return a list containing a single element.). The element of the list is a dict contraining the same keys with `heads`.
- Add your model in `model_factory` of `src/lib/models/model.py`.
