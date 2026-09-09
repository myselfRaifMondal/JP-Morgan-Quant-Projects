"""Task 2 - Price a natural gas storage contract (JPMorgan Forage).

``price_contract`` values a simple gas storage deal by walking the given
injection dates (buying at the market price, limited by ``injection_rate`` and
the remaining headroom under ``max_volume``) and then the withdrawal dates
(selling at the market price, limited by ``withdrawal_rate`` and the inventory
on hand), and finally subtracting ``storage_cost * max_volume``. Dates missing
from the price index are skipped, and the model ignores discounting and any
injection/withdrawal fees.

Input: ``load_gas_data(file_path)`` reads a CSV with a ``Date`` column (parsed
as the index) and a ``Price`` column. The example call at the bottom uses the
hard-coded path ``"Nat_Gas.csv"``, which is not in this repository. The bundled
series is ``datasets/naturalgas.csv``, whose columns are ``Dates`` and
``Prices``, so the path and the column names must be adjusted before running.

Output: prints the total contract value (net cash flow) for the hard-coded
example - two injections in Jan/Feb 2023, two withdrawals in Jun/Jul 2023,
rates of 1000 units, ``max_volume`` 5000 and ``storage_cost`` 10 per unit.
"""

import pandas as pd
from datetime import datetime

def load_gas_data(file_path):
    """Load natural gas price data from CSV."""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

def price_contract(data, injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_cost):
    """Calculate the value of the gas storage contract."""
    inventory = 0
    cash_flow = 0
    
    for date in injection_dates:
        if date in data.index:
            price = data.loc[date, 'Price']
            inject_volume = min(injection_rate, max_volume - inventory)
            inventory += inject_volume
            cash_flow -= inject_volume * price  # Buying cost
    
    for date in withdrawal_dates:
        if date in data.index:
            price = data.loc[date, 'Price']
            withdraw_volume = min(withdrawal_rate, inventory)
            inventory -= withdraw_volume
            cash_flow += withdraw_volume * price  # Selling revenue
    
    total_storage_cost = storage_cost * max_volume
    cash_flow -= total_storage_cost  # Deduct storage costs
    
    return cash_flow

# Example usage
data = load_gas_data("Nat_Gas.csv")
injection_dates = [datetime(2023, 1, 31), datetime(2023, 2, 28)]
withdrawal_dates = [datetime(2023, 6, 30), datetime(2023, 7, 31)]
injection_rate = 1000  # Example units per month
withdrawal_rate = 1000
max_volume = 5000
storage_cost = 10  # Example storage cost per unit

contract_value = price_contract(data, injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_cost)
print(f"Total contract value: {contract_value}")

