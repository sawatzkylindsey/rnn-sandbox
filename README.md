### Neural Network Weight Debugger (nn-wd) Visualization Tool
Tool to debug neural networks, specifically as their weights flow through the computational graph.

#### Sketch
<img src="sketch.jpg"/>

### Development
Requirements:

    # Python 3
    python3 -m venv p3
    source ./p3/bin/activate
    # Libraries (pip based)
    pip install matplotlib
    pip install nltk
    pip install numpy
    pip install sklearn
    pip install tensorflow==1.9.0
    # If above doesn't work (you'll have to manually fix the install tf library):
    #   pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.9.0-py3-none-any.whl
    # Libraries (custom)
    curl -LO https://github.com/sawatzkylindsey/pytils/archive/master.zip
    cd pytils-master/
    make install

Running:

    python dev-server.py -v

Running after system restart:

    python3 -m venv p3
    source ./p3/bin/activate
    python dev-server.py -v

