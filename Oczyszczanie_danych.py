from dateutil.parser import parse
from typing import List, NamedTuple, Optional, Counter
import datetime
import re
import pandas as pd


df = pd.read_csv("comma_delimited_stock_prices.csv", sep="/")
df.columns = ['symbol', 'date', 'closing_price']

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['AAPL', 'CDPR', 'NVD', "BTC"]
    
    def show_date(self): 
        return self.date


def parse_row(row: List[str])->StockPrice:
    symbolx, date, closing_price = row
    return StockPrice(symbol=symbolx,
                      date=parse(date).date(),
                      closing_price=float(closing_price)
                      )


def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row 
    if not re.match(r"^[A-Z]+$", symbol):
        return None  

    try:
        date = parse(date_).date()  
    except ValueError:
        return None  
    
    try:
        closing_price = float(closing_price_) 
    except ValueError:
        return None  
    
    return StockPrice(symbol, date, closing_price)  
for index, row in df.iterrows():

    row_data = [row['symbol'], row['date'], row['closing_price']]
    stock_price = try_parse_row(row_data)
    if not stock_price:
        print(f"Invalid data at index {index}: {row_data}")

# class calc(NamedTuple):
#     a:float
#     b:float
#     c:float
#     def add(self)->float:
#         return self.a + self.b +self.c
    
#     def div(self, d:int)->float:
#         return self.a - self.b -self.c - d
    
# class_calc = calc(1.2, 2.2, 3.2)
# print(class_calc.add())
# print(class_calc.div(1))

df = pd.read_csv('stocks.csv', sep=',')
# Symbol, Date,Open,High,Low,Close,Adj_Close,Volume =df["Symbol"],df["Date"],df["Open"],df["High"],df["Low"],df["Close"],df["Adj Close"],df['Volume']

filtred_data = df[df["Symbol"] == "AAPL"]
max_value = filtred_data["Close"].idxmax()
max_row = filtred_data.loc[max_value]

print(max_row)
