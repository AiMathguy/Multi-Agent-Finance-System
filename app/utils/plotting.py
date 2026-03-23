import matplotlib.pyplot as plt
import os

def save_plot(df, ticker):
    path = f"outputs/{ticker}_plot.png"

    plt.figure()
    plt.plot(df["Close"])
    plt.title(ticker)
    plt.savefig(path)
    plt.close()

    return path