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

- ETL + SQLite
- Baseline: same weekday + same hour average
- Models: Gradient Boosting (regression + classification)
- Features: hour, day of week, park_id, district
- GenAI: suggestion text generation

### 5. Analysis and Discussion

- Results: MAE, F1
- Visualizations: trends, heatmap
- Insights: ...

### 6. Challenges and Limitations

- Limited data collection period
- Collection gaps (notebook dependency)
- Not real-time on-street meters
- Weather features not yet fully integrated

### 7. Future Work

- Longer data collection
- Weather + holiday features
- Better interface / dashboard
- ...

### 8. Conclusion

### 9. References

### 10. Distribution of Work
