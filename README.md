# Pandas-Free Combinatorial Purged K-Fold Cross Validation

Sometimes you want to implement Combinatorial Purged Cross Validation. If you Google it, you can find many good implementations, but all of them are hard-coded with Pandas, leaving no room to choose another library for dataframes. It is troublesome if you cannot use or do not want to use Pandas.

Implementations found on Google:
- https://github.com/sam31415/timeseriescv
- https://www.kaggle.com/code/treename/janestreet-cv-method-combpurgedkfoldcv
- https://zenn.dev/ymd/articles/fd08fb46bc868c
- https://zenn.dev/sunbluesome/articles/0eaa8eea8375dd

I prefer to avoid using Pandas as much as possible, so it was a problem for me. Even when I asked ChatGPT, it came back with methods using Pandas, and I implemented it myself. Refer to this code if you want to use Combinatorial Purged Cross Validation but prefer to manipulate dataframes with Polars or such. The only third-party library this uses is Numpy.

## TODO
- [ ] Datetime splitting: It has not been implemented, but it is possible to do so using only the Python standard datetime library without using the dt-ish methods from Pandas or Polars.

## Usage
```
DATE = "date"
NUM_FOLDS = 6
NUM_TEST_FOLDS = 2
ONE_WEEK = 7
ONE_MONTH = 30

num_ticks = df[DATE].n_unique()

cpcv = CombinatorialPurgedKFold(num_ticks,
                                NUM_FOLDS,
                                NUM_TEST_FOLDS,
                                embargo_days=[ONE_WEEK, ONE_MONTH])
test_map, embargo_map = cpcv()

def extract_dates_from_index(index: int):
    return self.df[DATE].unique().sort().filter(pl.Series("", index))

for i, (test_idx, embargo_idx) in enumerate(zip(self.test_map.T, self.embargo_map.T)):
    print(f"Simulation {i}")

    test_dates = extract_dates_from_index(test_idx)
    embargo_dates = extract_dates_from_index(embargo_idx)

    train_df = df.filter(~pl.col(DATE).is_in(test_dates))\
        .filter(~pl.col(DATE).is_in(embargo_dates)).clone()
```
