# Think Bayes-Inspired Bayesian Programming Problems

These extension problems are inspired by the computational topic sequence in Allen B. Downey's *Think Bayes*, 2nd edition. They are original practice prompts written for this repository, not copied from the book.

The goal is to add a Bayesian-inference track that emphasizes practical grid methods in Python: priors, likelihoods, normalization, Bayes factors, sequential updating, conjugate models, posterior predictive distributions, joint distributions, mixtures, decision analysis, and simulation-style reasoning.

## Implementation structure

```text
solutions/think_bayes.py
tests/test_think_bayes.py
```

## Problems

### PMFs and Bayesian updating

1. **Normalize weights**

   Write a function that accepts a mapping of hypotheses to nonnegative weights and returns a normalized probability mass function.

2. **Uniform prior**

   Write a function that accepts a sequence of hypotheses and returns a uniform prior over the unique hypotheses.

3. **Bayesian PMF update**

   Write a function that accepts a prior PMF and likelihoods for each hypothesis, then returns the normalized posterior PMF.

4. **Marginal likelihood**

   Write a function that computes the total probability of the data under a prior and likelihood table.

### Odds and Bayes rule

5. **Probability and odds conversion**

   Write functions that convert probability to odds and odds to probability.

6. **Diagnostic-test posterior**

   Write a function that computes the posterior probability of a condition after a positive diagnostic test using prevalence, sensitivity, and false positive rate.

7. **Bayes factor**

   Write a function that computes the Bayes factor comparing two hypotheses from their likelihoods.

### Grid approximation

8. **Binomial likelihood grid**

   Write a function that returns likelihoods for observing a number of successes and failures across a grid of success probabilities.

9. **Binomial grid posterior**

   Write a function that updates a grid prior for a binomial success probability after observed successes and failures.

10. **Posterior summary**

    Write a function that returns posterior mean, MAP estimate, and cumulative probability at or below a requested threshold.

11. **Credible interval**

    Write a function that returns a central credible interval from a discrete PMF.

12. **Posterior predictive binary probability**

    Write a function that computes the posterior predictive probability of success for a Bernoulli trial.

### Conjugate models

13. **Beta-binomial posterior**

    Write a function that updates Beta prior parameters after observing successes and failures.

14. **Beta-binomial predictive probability**

    Write a function that computes the posterior predictive probability of success under a Beta-binomial model.

15. **Gamma-Poisson posterior**

    Write a function that updates Gamma prior parameters for a Poisson rate after observing event counts across one or more exposure units.

16. **Gamma-Poisson predictive mean**

    Write a function that computes the posterior predictive mean count for a new exposure period.

### Count and rate models

17. **Poisson rate grid update**

    Write a function that updates a discrete prior over Poisson rates after observing one or more counts.

18. **Poisson posterior predictive PMF**

    Write a function that mixes Poisson predictive distributions across a posterior PMF over rates.

19. **Normal mean grid update**

    Write a function that updates a discrete prior over possible normal means with known standard deviation.

20. **Normal posterior predictive grid**

    Write a function that approximates a posterior predictive distribution for a future normal observation by mixing across possible means.

### Joint distributions and mixtures

21. **Joint prior from independent PMFs**

    Write a function that builds a joint distribution from two independent PMFs.

22. **Marginalize a joint distribution**

    Write a function that marginalizes a joint distribution over one selected variable.

23. **Condition a joint distribution**

    Write a function that conditions a joint distribution on one selected variable and returns the posterior PMF for the other variable.

24. **Mixture PMF**

    Write a function that combines component PMFs using a PMF of mixture weights.

25. **Distribution of a sum**

    Write a function that computes the distribution of the sum of two independent discrete random variables.

26. **Distribution of a maximum**

    Write a function that computes the distribution of the maximum of two independent discrete random variables.

### Decision analysis and sequential inference

27. **Expected-loss table**

    Write a function that computes expected loss for each action under a posterior PMF and a supplied loss function.

28. **Minimum expected-loss action**

    Write a function that returns the action with the smallest expected loss.

29. **Sequential Bayesian updates**

    Write a function that repeatedly updates a posterior PMF from a sequence of likelihood tables.

30. **Grid update for mark-and-recapture**

    Write a function that estimates a population size on a grid after marking some individuals, sampling again, and counting recaptures.
