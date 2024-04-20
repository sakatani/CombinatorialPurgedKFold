# Pandas-Free Combinatorial Purged K-Fold Cross Validation

Sometimes you want to implement Combinatorial Purged Cross Validation, right? If you Google it, you can find many good implementations, but all of them are hard-coded with Pandas, leaving no room to choose another library for dataframes. It is troublesome if you cannot use or do not want to use Pandas.

Implementations found on Google:
- https://github.com/sam31415/timeseriescv
- https://www.kaggle.com/code/treename/janestreet-cv-method-combpurgedkfoldcv
- https://zenn.dev/ymd/articles/fd08fb46bc868c
- https://zenn.dev/sunbluesome/articles/0eaa8eea8375dd

I prefer to avoid using Pandas as much as possible, so it was a problem for me. Even when I asked ChatGPT, it came back with methods using Pandas, and I ended up implementing it myself. If you want to use Combinatorial Purged Cross Validation but prefer to manipulate dataframes with Polars or such, refer to this code. The only third-party library this uses is Numpy.

## TODO
- [ ] Datetime splitting: It has not been implemented, but it is possible to do so using only Python standard datetime library without using the dt-ish methods from Pandas or Polars.
