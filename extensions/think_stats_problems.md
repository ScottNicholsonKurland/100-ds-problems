# Think Stats-Inspired Statistical Programming Problems

These extension problems are inspired by the topic sequence in Allen B. Downey's *Think Stats*, 3rd edition. They are original practice prompts written for this repository, not copied from the book.

The goal is to add a statistics-heavy track that emphasizes practical data analysis in Python: exploratory data analysis, distributions, PMFs, CDFs, modeling distributions, relationships between variables, estimation, hypothesis testing, regression, time series, survival analysis, and analytic methods.

## Suggested implementation structure

```text
solutions/think_stats.py
tests/test_think_stats.py
```

## Problems

### Exploratory data analysis

1. **Column summary report**

   Write a function that accepts a pandas DataFrame and returns a summary DataFrame with one row per column. Include dtype, count, missing count, missing fraction, number of unique values, and example values.

2. **Group comparison summary**

   Write a function that compares a numeric variable across groups. Return count, mean, median, standard deviation, minimum, maximum, and interquartile range for each group.

### Distributions

3. **Frequency table from a Series**

   Write a function that accepts a pandas Series and returns a frequency table with columns for value, count, and proportion. The result should be sorted by descending count.

4. **Histogram bin counts**

   Write a function that accepts numeric data and a list of bin edges, then returns a DataFrame with bin labels, counts, and proportions.

### Probability mass functions

5. **Build a PMF**

   Write a function that accepts a sequence of discrete values and returns a probability mass function represented as a dictionary mapping each value to its probability.

6. **PMF mean and variance**

   Write functions that compute the mean and variance of a discrete distribution represented as a PMF dictionary.

7. **PMF comparison**

   Write a function that accepts two PMFs over the same outcome space and returns a DataFrame with each outcome, both probabilities, and the probability difference.

### Cumulative distribution functions

8. **Build a CDF**

   Write a function that accepts numeric data and returns a sorted DataFrame containing values and cumulative probabilities.

9. **Percentile rank**

   Write a function that computes the percentile rank of a value relative to a numeric sample.

10. **Value at percentile**

    Write a function that returns the value at a requested percentile from a numeric sample.

### Modeling distributions

11. **Normal model fit**

    Write a function that estimates the mean and standard deviation of a numeric sample, then returns expected quantiles from a fitted normal model.

12. **Exponential model fit**

    Write a function that estimates the rate parameter of an exponential distribution from observed waiting times. Include validation that all waiting times are nonnegative.

13. **Compare empirical and model CDFs**

    Write a function that compares an empirical CDF with a model CDF evaluated at the same data values and returns the maximum absolute difference.

### Probability density functions

14. **Kernel density estimate grid**

    Write a function that accepts numeric data and returns a DataFrame with x-values and estimated density values using a Gaussian kernel density estimate.

15. **Density normalization check**

    Write a function that accepts x-values and density values, approximates the area under the density curve, and returns whether it is close to 1 within a tolerance.

### Relationships between variables

16. **Covariance and correlation from scratch**

    Write functions that compute covariance and Pearson correlation from two equal-length numeric arrays without calling pandas or NumPy covariance helpers.

17. **Spearman correlation by ranks**

    Write a function that computes Spearman rank correlation by ranking two arrays and then computing Pearson correlation on the ranks.

18. **Conditional mean table**

    Write a function that bins one numeric variable and computes the mean of another numeric variable within each bin.

### Estimation

19. **Bootstrap confidence interval**

    Write a function that estimates a confidence interval for a statistic by bootstrap resampling. The statistic should be passed in as a function.

20. **Standard error by simulation**

    Write a function that repeatedly samples from a population array, computes a statistic on each sample, and returns the simulated standard error.

### Hypothesis testing

21. **Permutation test for difference in means**

    Write a function that performs a two-sided permutation test for the difference in means between two groups.

22. **Chi-square goodness-of-fit statistic**

    Write a function that computes the chi-square statistic for observed and expected category counts. Validate that expected counts are positive.

23. **Power simulation for a two-group test**

    Write a function that simulates the power of a two-group test as a function of effect size, sample size, and noise level.

### Least squares

24. **Simple least-squares fit from scratch**

    Write a function that computes the intercept and slope of a simple linear regression using closed-form formulas.

25. **Residual diagnostics table**

    Write a function that accepts x-values, y-values, intercept, and slope, then returns a DataFrame with fitted values, residuals, absolute residuals, and squared residuals.

### Multiple regression

26. **Design matrix builder**

    Write a function that accepts a DataFrame, a list of numeric feature columns, and a flag for adding an intercept column. Return a NumPy design matrix.

27. **Multiple regression with categorical encoding**

    Write a function that one-hot encodes selected categorical columns, combines them with numeric columns, and fits a linear regression model.

### Time series and survival analysis

28. **Survival curve from event data**

    Write a function that accepts durations and event indicators, then returns a simple Kaplan-Meier-style survival table with time, number at risk, number of events, and survival probability.
