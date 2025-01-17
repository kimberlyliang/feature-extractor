# BMIN 5100 Template Repository
Template repository for BMIN 5100 Projects.

Contains a minimal Python project that reads a CSV from an input directory and
outputs a CSV to an output directory, suitable for an analysis workflow on Pennsieve.

Use this template to create your own GitHub repository by clicking the green
`Use this template` button towards the top-right of this page.

### Setup
Install the following:
- `python3` (latest)
- `pip` (or `pip3`, latest)

Then, run the following
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Running the application
```
python3 main/app.py
```

### Testing the application
```
pytest
```
