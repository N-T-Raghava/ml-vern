Developer Guide
===============

How to contribute, run tests, and build the project locally.

Local development
-----------------

1. Create and activate a virtual environment::

   python -m venv .venv
   .venv\Scripts\activate

2. Install dev dependencies::

   pip install -e .[dev]

Running tests
-------------

Use `pytest` to run the test suite. See `pytest.ini` for configuration.

Docs build
----------

To build the docs locally::

   sphinx-build -b html docs/source docs/build/html
