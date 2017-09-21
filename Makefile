
make install:
    pip install conda
	conda update -q conda
    conda env create -f environment.yml
    source activate Replay-Content-Classification
	python setup.py develop
