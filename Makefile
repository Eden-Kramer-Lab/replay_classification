
make install:
    pip install conda
	conda update -q conda
    conda env create -f environment.yml
    source activate replay_classification
	python setup.py develop
