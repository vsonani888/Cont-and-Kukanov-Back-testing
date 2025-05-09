# Cont-and-Kukanov-Back-testing

Simulation of a smart order routing strategy using the static cost model from Cont and Kukanov's "Optimal Order Placement in Limit Order Markets". Goal is to split 5000 shares across multiple venues to reduce the total cost.

## Overview of backtest.py

- Extracts the l1_day.csv to get the prices and sizes for the venue over a period of time
- Implements a grid function that parses over all of the permuations of ranges for the lambda over, under and the theta queue values with steps. 
- During the permutations, it uses the backtest to find the best cost which later updates all of the best values that will be used for the results
- Backtest uses allocate function to find the split value of how much share it needs to buy at the certain venue.
- The results from the optimal permutations are compared with the twap, vwap and the bast ask values. 
- Best ask buys from the venue with the lowest ask price, Twap spilts it between equal buckets, and Vwap splits it depending on the available size per venue.


## Output

- The best parameter for the lambda and theta are found and outputed for the backtest which can be used later for different tests and implementation.
- The total cost and the avg price of the test data given.
- comparision with the benchmark twap, vwap and the best ask.
- Generates a results.png file which shows how the cost change over time

## Running the file

```bash
python backtest.py