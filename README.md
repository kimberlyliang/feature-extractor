# Template Repository

This repository contains code for processing iEEG data and extracting various features.

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd template
```

2. Build and run with Docker:
```bash
docker-compose up --build
```

## Data Structure

- Input data should be placed in `data/input/`
- Output will be generated in `data/output/`

## Features

The code extracts the following features from iEEG data:
- Catch22 features
- FOOOF features
- Bandpower features
- Entropy features

## Environment Variables

The following environment variables can be set:
- `INPUT_DIR`: Directory containing input data (default: /data/input)
- `OUTPUT_DIR`: Directory for output data (default: /data/output)
- `ENVIRONMENT`: Set to 'LOCAL' for local development
