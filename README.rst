.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


===============
Neural Operator(Paddle backend)
===============

.. image::  doc/_static/paddle_logo.png

.. important::
   
   This branch (paddle) experimentally integrates the `Paddle backend <https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html>`_ to Neural Operator.

   It was developed from version 0.3.0 of Neural Operator. It is recommended to install **nightly-build (develop)** Paddle before running any code in this branch.

   It was verified on Ubuntu 20.04. It may encounter some problems if you are using another environment.

Installation
------------

.. code::

   # install nightly-build paddlepaddle
   python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

   # triton
   pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

   # install paddle_harmonics
   git clone https://github.com/PFCCLab/paddle_harmonics.git
   cd paddle_harmonics
   pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

   # install nightly-build ppsci
   python -m pip install https://paddle-qa.bj.bcebos.com/PaddleScience/whl/latest/dist/paddlesci-0.0.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

   # install neuraloperator
   cd neuraloperator
   pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

Unit Test
------------

.. code::

   cd neuraloperator/example
   pytest

Below is Neural Operator's original README
===============

``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.


Installation
------------

Just clone the repository and install locally (in editable mode so changes in the code are immediately reflected without having to reinstall):

.. code::

  git clone https://github.com/NeuralOperator/neuraloperator
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt

You can also just pip install the library:


.. code::
  
  pip install neuraloperator

Quickstart
----------

After you've installed the library, you can start training operators seemlessly:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, out_channels=1)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized',
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!

Checkout the `documentation <https://neuraloperator.github.io/neuraloperator/dev/index.html>`_ for more!

Using with weights and biases
-----------------------------

Create a file in ``neuraloperator/config`` called ``wandb_api_key.txt`` and paste your Weights and Biases API key there.
You can configure the project you want to use and your username in the main yaml configuration files.

Contributing code
-----------------

All contributions are welcome! So if you spot a bug or even a typo or mistake in
the documentation, please report it, and even better, open a Pull-Request on 
`GitHub <https://github.com/neuraloperator/neuraloperator>`_. Before you submit
your changes, you should make sure your code adheres to our style-guide. The
easiest way to do this is with ``black``:

.. code::

   pip install black
   black .

Running the tests
=================

Testing and documentation are an essential part of this package and all
functions come with uni-tests and documentation. The tests are ran using the
pytest package. First install ``pytest``:

.. code::

    pip install pytest
    
Then to run the test, simply run, in the terminal:

.. code::

    pytest -v neuralop
    
Citing
------

If you use NeuralOperator in an academic paper, please cite [1]_, [2]_::

   @misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
   }

   @article{kovachki2021neural,
      author    = {Nikola B. Kovachki and
                     Zongyi Li and
                     Burigede Liu and
                     Kamyar Azizzadenesheli and
                     Kaushik Bhattacharya and
                     Andrew M. Stuart and
                     Anima Anandkumar},
      title     = {Neural Operator: Learning Maps Between Function Spaces},
      journal   = {CoRR},
      volume    = {abs/2108.08481},
      year      = {2021},
   }


.. [1] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar A., “Fourier Neural Operator for Parametric Partial Differential Equations”, ICLR, 2021. doi:10.48550/arXiv.2010.08895.

.. [2] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., “Neural Operator: Learning Maps Between Function Spaces”, JMLR, 2021. doi:10.48550/arXiv.2108.08481.
