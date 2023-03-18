# Predicting Economic Variables

This project uses deep learning to design a forcasting network to predict future values of different [economic variables](https://economicpoint.com/economic-variables#:~:text=Economic%20variables%20are%20measurements%20that,characteristics%20that%20describe%20an%20object.https:/) . It finds some correlated variables based on past data and then uses trains different neural networks for predict the future variables data from its correlated variable.

For e.g

Input to network:

* Main Variable: Values from May 20 to Oct 20
* Correlated Variables: Values from May 20 to Nov 20

Output:

* Main Variable value at Nov 20

## Installation ( tested on Python 3.10 and conda)

* Activate new conda environment with python 3.10

`conda env create --name envname --file=environments.yml`

* Run Training

`python src/train.py`

* Run Evaluation

`python src/evaluate.py`

NOTE: Configuration files are stored in `var/`
