import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    file_path = 'l1_day.csv'
    order_size =  5000

    total_cost = 0.0
    total_shares = 0
    lambda_over = 0.01
    lambda_under = 0.02
    theta_queue = 0.001

    data = read_file(file_path)

    remaining_size = order_size

    for ts, venues in data:

        split = allocate(remaining_size, venues, lambda_over, lambda_under, theta_queue)

        if not split:
            continue

        for i in range(len(venues)):

            qty = split[i]
            ask = venues[i]['ask']
            fee = venues[i]['fee']
            rebate = venues[i]['rebate']
            ask_size = venues[i]['ask_size']
            
            exe = min(qty, ask_size)
            cost = exe * (ask + fee)
            rebate = max(qty - exe, 0) * rebate

            total_cost += cost - rebate
            total_shares += exe
            remaining_size -= exe

    avg_price = total_cost/ total_shares

    print("avg price", avg_price)