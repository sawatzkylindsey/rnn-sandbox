### Rnn Sandbox
A sandbox RNN-Visualization environment for my research.
This branch shows our IEEE VIS 2019 submission (https://arxiv.org/abs/1908.00588):

> Visualizing RNN States with Predictive Semantic Encodings

> Lindsey Sawatzky, Steven Bergner, Fred Popowich

> Simon Fraser University

### 1-Command Usage

    # After setting up the requirements, run the following command
    ./pipeline.sh <NATURAL_LANGUAGE_TEXT_FILE> <OUTPUT_PREFIX>
    # View the server at http://localhost:8888/index.html

### Requirements

    # Python 3
    python3 -m venv p3
    source ./p3/bin/activate
    # Libraries (pip based)
    pip install matplotlib
    pip install nltk
    pip install numpy
    pip install sklearn
    pip install sympy
    pip install tensorflow==1.9.0
    # If above doesn't work (you'll have to manually fix the install tf library):
    #   pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.9.0-py3-none-any.whl
    # Libraries (custom)
    curl -LO https://github.com/sawatzkylindsey/pytils/archive/master.zip
    cd pytils-master/
    make install

### Development
Setup Data:

    ...

Running:

    python dev-server.py -v

Running after system restart:

    python3 -m venv p3
    source ./p3/bin/activate
    python dev-server.py -v

