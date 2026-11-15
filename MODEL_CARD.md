# Model card: CardioRisk v1.0

**Model.** L2-regularised logistic regression over 13 clinical variables with median imputation
and missing-indicator features. Output: probability of angiographic coronary artery disease.

**Intended use.** Education and research on clinical prediction modelling.

**Out of scope.** Diagnosis, triage or any decision about a real patient. Screening of people
who have not been referred for cardiac evaluation.

**Training data.** UCI Heart Disease, 920 patients from four hospitals in the USA, Hungary and
Switzerland, collected in the 1980s. 79% male. No race or ethnicity information.

**Performance.** Internal cross-validated AUROC 0.903 (95% CI 0.882 to 0.922). Mean AUROC on a
hospital held out of training 0.812 (range 0.723 to 0.904). Details in `reports/RESULTS.md`.

**Known failure modes.**
- Predicted risks are over-confident at hospitals with a different case mix (calibration slope
  as low as 0.49).
- Sensitivity at the default threshold is lower in women (74.0%) than men (91.9%).
- Performance is weaker in patients aged 55 and over (AUROC 0.851) than under 55 (0.924).

**Ethical considerations.** A model trained mostly on men from three countries, forty years ago,
should not be assumed to generalise to other populations, including Indian patients, in whom
CAD tends to present younger. Local validation is required before any real-world use.
