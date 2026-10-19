# Capstone Report Outline

## Hong Kong Parking Vacancy Prediction

### 1. Group Details

- Group Number / Name
- Members

### 2. Project Details

- Title: Predicting Parking Vacancy in Kwun Tong and Sha Tin (30-60 min ahead)
- Objectives: ...

### 3. Dataset Description

- Source: Transport Department Parking Vacancy API
- Scope: ~149 car parks in 觀塘區 + 沙田區
- Characteristics: Time series, vacancy counts, collected every ~10 min
- Preprocessing: ...

### 4. Methodology

- Data collection: Automated polling of Transport Department API every ~10 minutes, stored in SQLite (`vacancy_records` and `carparks` tables).
- Scope: 149 car parks in 觀塘區 and 沙田區, private cars only.
- Baseline: Historical average vacancy by park_id + hour + day of week.
- Models:
  - Gradient Boosting Regressor for vacancy count prediction
  - Gradient Boosting Classifier for has_vacancy (vacancy > 0)
- Features used: hour, day of week, is_weekend, park_id (encoded), district (encoded)
- Evaluation: Time-based train/test split (70/30), MAE for regression, F1 for classification.
- GenAI component: Rule-based suggestion text generation (to be enhanced with LLM later).

### 5. Analysis and Discussion

- Current data: ~3100+ records over 2 days.
- Regression test MAE: ~10.73 (better than baseline ~17.84)
- Classification F1: ~0.957
- Visualizations produced: vacancy trend lines, district-hour heatmap, top-5 recommendation bar chart.
- Observation: Night-time average vacancy higher than afternoon; Sha Tin shows higher night vacancy than Kwun Tong in current data.

### 6. Challenges and Limitations

- **Limited temporal coverage**: Only ~2 days of data so far. Patterns may not generalize to weekdays, holidays, or different weather conditions.
- **Collection gaps**: Data collection depends on a continuously running Jupyter notebook. Overnight or when the computer sleeps, gaps appear (visible in the heatmap).
- **Scope limitation**: Only car park vacancy (not individual on-street metered spaces). Results should not be claimed as covering all parking in the districts.
- **Feature completeness**: Weather data has been tested but not yet systematically joined into the feature table for model training.
- **Model simplicity**: Current models use basic time and location features. More advanced time-series models or external event data were not included due to time constraints.
- **Evaluation**: With short data period, the time-based split still has limited diversity; longer collection will improve reliability of MAE and F1.

### 7. Future Work

- Longer data collection
- Weather + holiday features
- Better interface / dashboard
- ...

### 8. Conclusion

### 9. References

### 10. Distribution of Work
