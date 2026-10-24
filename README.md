# HK Parking Vacancy Prediction

Capstone project: Predict parking vacancy 30–60 minutes ahead for car parks in Kwun Tong and Sha Tin districts.

## Research Question

Can we predict whether a car park will have private-car vacancies (classification) and the number of vacancies (regression) 30–60 minutes ahead, using official real-time data + time features, and outperform a same-weekday-same-hour historical average baseline?

## Current Status

- Data collection running (every ~10 min) via `collect.ipynb`
- SQLite database with `vacancy_records` and `carparks` tables
- ~150 car parks in 觀塘區 + 沙田區
- Baseline and Gradient Boosting models implemented
- Basic EDA visualizations and recommendation query function
- Report outline drafted

## Key Files

- `explore_api.ipynb`: Exploration, modeling, EDA, and functions
- `collect.ipynb`: Continuous data collection
- `parking.db`: SQLite database
- `app.py`: Streamlit interface
- `report_outline.md`: Report structure and current findings

## Next Steps

- Continue data collection
- Integrate weather features
- Build simple interface
- Complete final report and presentation
