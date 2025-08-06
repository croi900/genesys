# Genesys - Big Bang Nucleosynthesis Model Analysis

## Overview
Genesys is a Python framework for analyzing Big Bang Nucleosynthesis (BBN) models using the PRyMordial framework. It implements various potential models and uses Monte Carlo Chain (MCMC) methods for parameter estimation and analysis of primordial element abundances.

## Features
- MCMC analysis capabilities
- Integration with PRyMordial BBN calculations
- Database storage for results
- Parallel computation support using Dask
- Computation of primordial abundances (Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7)

## Requirements
- Python 3.x
- uv package manager
- Dependencies (automatically managed by uv):
  - numpy
  - scipy
  - schwimmbad
  - dask
  - dataset
  - pymysql
  - python-dotenv

## Quick Start
1. Clone the repository
```bash
git clone https://github.com/croi900/genesys.git
cd genesys
```

2. Install dependencies using uv
```bash
uv pip install .
```

3. Run the main script
```bash
uv run main.py
```

## Project Structure
- `main.py` - Entry point and MCMC configuration
- `models/` - Implementation of various potential models
- `PRyM/` - PRyMordial framework integration
- `db.py` - Database operations and result storage
- `ga.py` - Genetic Algorithm implementation
- `mcmc.py` - Monte Carlo Chain methods

## Database Configuration
The project uses MySQL for storing results. Set up your database credentials in `.env` file:
```env
DB_URL=mysql+pymysql://user:password@localhost/weyl_oop
```

## Contributors
- @croi900
- @teomatei22

