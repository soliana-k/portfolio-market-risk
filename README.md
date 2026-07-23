### Portfolio/Equity Market Risk 

## Phase 1: Data Collection



## 1. Project Overview

This repository contains the foundational data pipeline for the Equity Risk Platform. It handles automated data extraction from Yahoo Finance, computes financial return metrics, validates portfolio weights, and stores raw/processed artifacts securely under strict data integrity rules.

---

## 2. Project Directory Structure

```text
project/
├── src/
│   └── data_collection/
│       ├── download_portfolio.py   
│       ├── returns.py              
│       └── build_portfolio.py      
├── data/
│   └── raw/                        
├── tests/                         
├── .github/
│   └── workflows/                  
├── requirements.txt                
└── README.md

```

---

## 3. Installation & Setup

### Prerequisites

Clone the repository and install the approved dependencies:

```bash
pip install -r requirements.txt

```

### Dependencies
see 
```text
requirements.txt
```

---

## 4. Testing & Verification

Run the automated test suite using `pytest` to verify technical correctness and business logic across all modules:

```bash
pytest tests/ -v

```

### Test Results & Verification Screenshot

![alt text](image.png)
