import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

def read_file(file_path):
    df = pd.read_csv(file_path)

    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])
    print(df.head(), len(df))

    fee = 0.001
    rebate = 0.002

    data = []
    grouped = df.groupby('ts_event')

    for ts, group in grouped:
        venues = []

        for item, row in group.iterrows():
            
            venues.append({"publisher_id" : row['publisher_id'], "ask" : row['ask_px_00'], "ask_size" : row['ask_sz_00'], "fee" : fee, "rebate" : rebate})
    
        data.append((ts, venues))
    print(data[:5])

    return data
            
def allocate(remaining_size, venues, lambda_over, lambda_under, theta_queue):

    step = 100
    splits = [[]]

    for v in range(len(venues)):
        new_splits = []

        for alloc in splits:
            used = sum(alloc)
            max_v = min(remaining_size - used, venues[v]['ask_size'])

            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
            
            splits = new_splits

        best_cost = float('inf')
        best_split = []

        for alloc in splits:

            if sum(alloc) != remaining_size:
                continue

            cost = compute_cost(alloc, venues, remaining_size, lambda_over, lambda_under, theta_queue)

            if cost < best_cost:
                best_cost = cost
                best_split = alloc
            
    #return best_split, best_cost
    return best_split

def compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue):

    exec = 0
    cash_spent = 0.0

    for i in range(len(venues)):
        
        qty = alloc[i]
        ask = venues[i]['ask']
        fee = venues[i]['fee']
        rebate = venues[i]['rebate']
        ask_size = venues[i]['ask_size']

        exe = min(qty, ask_size)
        exec += exe
        cash_spent += exe * (ask * fee)
        maker_rebate = max(qty - exe, 0) * rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - exec, 0)
    overfill = max(exec - order_size, 0)

    risk_pen = theta_queue * (underfill + overfill)
    file_pen = (lambda_under * underfill) + (lambda_over * overfill)

    return cash_spent + risk_pen + file_pen

def grid_search(data, lo_range, lu_range, tq_range, lo_step, lu_step, tq_step, order_size):
    best_lo = None
    best_lu = None
    best_tq = None

    best_cost = float('inf')
    best_avg_price = 0
    best_curve = []

    #make array of all values by adding the step size

    lo_values = np.arange(lo_range[0], lo_range[1] + lo_step, lo_step)
    lu_values = np.arange(lu_range[0], lu_range[1] + lu_step, lu_step)
    tq_values = np.arange(tq_range[0], tq_range[1] + tq_step, tq_step)

    #runs it for each of the values 

    for lo in lo_values:
        for lu in lu_values:
            for tq in tq_values:
                cost, avg_price, curve = backtest(data, lo, lu, tq, order_size)

                if cost < best_cost:
                    best_cost = cost
                    best_avg_price = avg_price
                    best_lo = lo
                    best_lu = lu
                    best_tq = tq
                    best_curve = curve

    return best_lo, best_lu, best_tq, best_cost, best_avg_price, best_curve


def backtest(data, lo, lu, tq, order_size):

    remaining_size = order_size
    total_cost = 0.0
    total_shares = 0
    cost_over_time = []

    filled = False

    for ts, venues in data:

        if not filled:

            #gets the split value that we can use to find the value

            split = allocate(remaining_size, venues, lo, lu, tq)

            if split:

                for i in range(len(venues)):

                    qty = split[i]
                    ask = venues[i]['ask']
                    fee = venues[i]['fee']
                    rebate = venues[i]['rebate']
                    ask_size = venues[i]['ask_size']

                    #depending on the split we buy the shares while saving the costs and such
                    
                    exe = min(qty, ask_size)
                    cost = exe * (ask + fee)
                    rebate_amt = max(qty - exe, 0) * rebate

                    total_cost += cost - rebate_amt
                    total_shares += exe
                    remaining_size -= exe
                
                if remaining_size <= 0:
                    filled = True

        cost_over_time.append(total_cost)

    avg_price = total_cost / total_shares

    return total_cost, avg_price, cost_over_time

def baseline_best_ask(data, order_size):
    remaining_size = order_size
    total_cost = 0.0
    total_shares = 0

    for ts, venues in data:

        #basically buys the shares starting from the cheapest venue

        best = venues[0]

        for venue in venues:
            if venue['ask'] < best['ask']:
                best = venue

        qty = min(remaining_size, best['ask_size'])

        cost = qty * (best['ask'] + best['fee'])
        total_cost += cost
        total_shares += qty
        remaining_size -= qty

    avg_price = total_cost/total_shares

    return total_cost, avg_price

def baseline_twap(data, order_size):
    remaining_size = order_size
    total_cost = 0.0
    total_shares = 0

    #bascially divide up the order size into 9 blocks that would buy the shares

    block = 9
    block_size = order_size // block
    interval = len(data) // block

    for i in range(block):
        
        idx = i * interval
        venues = data[idx][1]
        share_left = block_size

        for v in venues:
            qty = min(share_left, v['ask_size'], remaining_size)
            cost = qty * (v['ask'] + v['fee'])

            total_cost += cost
            total_shares += qty
            remaining_size -= qty
            share_left -= qty

        #if it cannot be filled than we just force feed it

        j = idx + 1

        while share_left > 0 and j < len(data):
            for venue in data[j][1]:

                qty = min(share_left, v['ask_size'], remaining_size)
                cost = qty * (v['ask'] + v['fee'])

                total_cost += cost
                total_shares += qty
                remaining_size -= qty
                share_left -= qty
            
            j += 1


    avg_price = total_cost/total_shares

    #print("toal shares",  total_shares)

    return total_cost, avg_price


def baseline_vwap(data, order_size):
    remaining_size = order_size
    total_cost = 0.0
    total_shares = 0

    for ts, venues in data:

        #bascially buys shares depending on the number of shares each venue gives, splits it evenly

        total_vol = 0

        for v in venues:
            total_vol += v['ask_size']

        for venue in venues:

            share = venue['ask_size'] / total_vol
            qty = min(share * remaining_size, venue['ask_size'])

            cost = qty * (venue['ask'] + venue['fee'])
            total_cost += cost
            total_shares += qty
            remaining_size -= qty

    avg_price = total_cost / total_shares 

    return total_cost, avg_price
    
if __name__ == "__main__":
    file_path = 'l1_day.csv'
    order_size =  5000

    data = read_file(file_path)

    lo_range = (0.001, 0.01)
    lo_step = 0.002
    lu_range = (0.01, 0.1)
    lu_step = 0.02
    tq_range = (0.0001, 0.001)
    tq_step = 0.003

    #updates the best vales depending on the range of all values applied for each backtest

    best_lo, best_lu, best_tq, best_cost, best_avg_price, best_curve = grid_search(data, lo_range, lu_range, tq_range, lo_step, lu_step, tq_step, order_size)

    #some tests to get some benchmarks
    
    best_ask_cost, best_ask_avg = baseline_best_ask(data, order_size)
    twap_cost, twap_avg = baseline_twap(data, order_size)
    vwap_cost, vwap_avg = baseline_vwap(data, order_size)

    #output result in json format
    
    result = {

        "best_weights" : {"lambda_over" : best_lo, "lambda_under" : best_lu, "theta_queue" : best_tq},

        "best_cost" : best_cost,
        "best_avg_price" : best_avg_price,

        "best_ask_cost" : best_ask_cost,
        "best_ask_avg_price" : best_ask_avg,

        "twap_cost" : twap_cost,
        "twap_avg" : twap_avg,

        "vwap_cost" : vwap_cost,
        "vwap_avg" : vwap_avg,

        "savings_vs_best_ask" : (best_ask_cost - best_cost) / best_ask_cost,
        "savings_vs_twap" : (twap_cost - best_cost) / twap_cost,
        "savings_vs_vwap" : (vwap_cost - best_cost) / vwap_cost

    }

    #print(twap_cost," ",  best_cost)

    print(json.dumps(result))

    print(best_curve[:10])
    print(len(best_curve))

    plt.plot(best_curve)
    plt.savefig("results.png")

    #print("avg price", avg_price)