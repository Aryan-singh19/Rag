Read the project handout first.
Use the commands below to set up the environment and run the tests.

From the handout directory:
  conda env create -f environment.yml
  conda activate worldmodels
  cd starter_code
  python tests/smoke_test.py

After completing the TODOs:
  python -m pytest -q tests
