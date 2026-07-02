# Dataset

**Name:** ReneWind Generator Failure Dataset
**Source:** MIT IDSS coursework (not redistributable)
**Files:** Train.csv + Test.csv
**Size:** ~40,000 rows x 40+ anonymized sensor features
**Target:** 1 = failure, 0 = no failure
**Class imbalance:** ~5.5% failures

## Cost Structure
- True Positive (repair): lowest cost
- False Positive (unnecessary inspection): medium cost
- False Negative (missed failure = replacement): highest cost

Recall on the failure class is the primary metric.
