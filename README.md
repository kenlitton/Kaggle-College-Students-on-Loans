# Overview

The goal of this project is to accurately predict the % of students who have taken out a federal student loan from a number of universities across the country. To stay organized, ScikitLearn Pipelines were used in the script labeled 'regression_challenge.py' and the same script transformed incoming data and output a Kaggle ready csv file with predictions for a test data CSV (this was a private Kaggle competition among my Data Science Immersive classmates at General Assembly).

## Modeling

I bagged DecisionTreeRegressors together and the code can be found under the file titled regression_challenge.py. A number of steps were taken to find the optimum parameters and those steps can be found in the section on leveraging ScikitLearn's GridSearchCV function.

## Results

After tuning parameters and submitting to Kaggle I yielded a RMSE of 17.439 on the held-out test dataset and finished second in my class of 12 students.  

## Data

The variables in the data can be found as follows:

UNITID - The Unit ID for the Institution

INSTNM - Name of the Institution

CITY - City

STABBR - State

ZIP - Zip Code

ACCREDAGENCY - Accrediting Agency

MAIN - Dummy, is this the main campus

NUMBRANCH - Number of Branches (including main campus)

PREDDEG - Categorical, Predominant Degree Served

0: Not Classified
1: Predominantly Certificates
2: Predominantly Associates
3: Predominantly Bachelors
4: Entirely Graduate Degrees

HIGHDEG - Categorical, Highest Degree Awarded

0: Non-Degree
1: Certificate
2: Associates
3: Bachelors
4: Graduate Degrees

CONTROL - Categorical, Private/Public Status

1: Public
2: Private Non-Profit
3: Private For-Profit

LOCALE - Categorical, Type of Area for College

11 City: Large (population of 250,000 or more)
12 City: Midsize (population of at least 100,000 but less than 250,000)
13 City: Small (population less than 100,000)
21 Suburb: Large (outside principal city, in urbanized area with population of 250,000 or more)
22 Suburb: Midsize (outside principal city, in urbanized area with population of at least 100,000 but less than 250,000)
23 Suburb: Small (outside principal city, in urbanized area with population less than 100,000)
31 Town: Fringe (in urban cluster up to 10 miles from an urbanized area)
32 Town: Distant (in urban cluster more than 10 miles and up to 35 miles from an urbanized area)
33 Town: Remote (in urban cluster more than 35 miles from an urbanized area)
41 Rural: Fringe (rural territory up to 5 miles from an urbanized area or up to 2.5 miles from an urban cluster)
42 Rural: Distant (rural territory more than 5 miles but up to 25 miles from an urbanized area or more than 2.5 and up to 10 miles from an urban cluster)
43 Rural: Remote (rural territory more than 25 miles from an urbanized area and more than 10 miles from an urban cluster)

CCUGPROF -  Categorical, Carnegie Classification for Undergrad Profile

-2 Not applicable
0 Not classified (Exclusively Graduate)
1 Two-year, higher part-time
2 Two-year, mixed part/full-time
3 Two-year, medium full-time
4 Two-year, higher full-time
5 Four-year, higher part-time
6 Four-year, medium full-time, inclusive, lower transfer-in
7 Four-year, medium full-time, inclusive, higher transfer-in
8 Four-year, medium full-time, selective, lower transfer-in
9 Four-year, medium full-time , selective, higher transfer-in
10 Four-year, full-time, inclusive, lower transfer-in
11 Four-year, full-time, inclusive, higher transfer-in
12 Four-year, full-time, selective, lower transfer-in
13 Four-year, full-time, selective, higher transfer-in
14 Four-year, full-time, more selective, lower transfer-in
15 Four-year, full-time, more selective, higher transfer-in

CCSIZSET - Categorical, Carnegie Classification Size and Setting

-2 Not applicable
0 (Not classified)
1 Two-year, very small
2 Two-year, small
3 Two-year, medium
4 Two-year, large
5 Two-year, very large
6 Four-year, very small, primarily nonresidential
7 Four-year, very small, primarily residential
8 Four-year, very small, highly residential
9 Four-year, small, primarily nonresidential
10 Four-year, small, primarily residential
11 Four-year, small, highly residential
12 Four-year, medium, primarily nonresidential
13 Four-year, medium, primarily residential
14 Four-year, medium, highly residential
15 Four-year, large, primarily nonresidential
16 Four-year, large, primarily residential
17 Four-year, large, highly residential
18 Exclusively graduate/professional

HBCU - Dummy, Historically Black College

PBI - Dummy, Predominantly Black College

MENONLY - Dummy, Men-Only College

WOMENONLY - Dummy, Women-Only College

DISTANCEONLY - Dummy, Distance Only Campus

UGDS - Enrollment of Undergrad degree-seekers

AGE_ENTRY - Age of Entry via Social Security Administration

FEMALE - Share of Female Students via Social Security Administration

MARRIED - Share of Married Students

DEPENDENT - Share of Dependent Students (for tax purposes)

MD_FAMINC - Median Family Income in real 2015 Dollars

percent_on_student_loan - Percent of students on federal student loans

id_number - ID number for submission
