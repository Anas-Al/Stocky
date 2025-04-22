import discord
from discord import app_commands
from discord.ext import commands
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

@bot.event
async def on_ready():
    await tree.sync()
    print(f"✅ Bot is online as {bot.user}")

@tree.command(name="stock", description="Get VWAP, RSI(5), and MACD(6,13,5) for a stock")
@app_commands.describe(ticker="Stock symbol (e.g. AAPL, TSLA, SPY)")
async def stock(interaction: discord.Interaction, ticker: str):
    try:
        await interaction.response.defer()
    except discord.NotFound:
        await interaction.channel.send("⚠️ This command took too long. Please try again.")
        return

    ticker = ticker.upper()

    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=False)

        if data.empty or not all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
            await interaction.followup.send(f"❌ Could not retrieve valid data for `{ticker}`.")
            return

        data = data[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors='coerce')
        data.dropna(inplace=True)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        data["VWAP"] = (tp * data["Volume"]).cumsum() / data["Volume"].cumsum()

        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=5).mean()
        avg_loss = loss.rolling(window=5).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        ema6 = data["Close"].ewm(span=6, adjust=False).mean()
        ema13 = data["Close"].ewm(span=13, adjust=False).mean()
        macd = ema6 - ema13
        signal = macd.ewm(span=5, adjust=False).mean()

        price = data["Close"].iloc[-1].item()
        vwap_now = data["VWAP"].iloc[-1].item()
        rsi_now = rsi.dropna().iloc[-1].item()
        macd_now = macd.dropna().iloc[-1].item()
        signal_now = signal.dropna().iloc[-1].item()

        if price > vwap_now and macd_now > signal_now and rsi_now > 30:
            trend = "📈 Bullish"
        elif price < vwap_now and macd_now < signal_now and rsi_now < 70:
            trend = "📉 Bearish"
        else:
            trend = "⚠️ Neutral"

        embed = discord.Embed(
            title=f"{ticker} Technical Analysis",
            description=f"{trend}\n\n**VWAP**, **RSI(5)**, **MACD(6,13,5)** based on daily candles.",
            color=discord.Color.green() if "Bullish" in trend else discord.Color.red() if "Bearish" in trend else discord.Color.orange()
        )
        embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
        embed.add_field(name="VWAP", value=f"${vwap_now:.2f}", inline=True)
        embed.add_field(name="RSI (5)", value=f"{rsi_now:.2f}", inline=True)
        embed.add_field(name="MACD", value=f"{macd_now:.4f}", inline=True)
        embed.add_field(name="Signal", value=f"{signal_now:.4f}", inline=True)
        embed.set_footer(text="Data from Yahoo Finance | VWAP, RSI, MACD")
        embed.timestamp = discord.utils.utcnow()

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Error fetching data for `{ticker}`:\n```{str(e)}```")

@tree.command(name="option", description="Estimate ITM, POP, and Greeks for a stock option")
@app_commands.describe(
    ticker="Stock symbol (e.g. TSLA)",
    strike="Strike price (e.g. 200)",
    expiration="Expiration date (YYYY-MM-DD)",
    type="Option type: Call or Put"
)
@app_commands.choices(type=[
    app_commands.Choice(name="Call", value="call"),
    app_commands.Choice(name="Put", value="put")
])
async def option(
    interaction: discord.Interaction,
    ticker: str,
    strike: float,
    expiration: str,
    type: app_commands.Choice[str]
):
    try:
        await interaction.response.defer()
    except discord.NotFound:
        await interaction.channel.send("⚠️ This command took too long. Try again.")
        return

    try:
        ticker = ticker.upper()
        option_type = type.value
        exp_date = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
        today = datetime.date.today()
        T = (exp_date - today).days / 365.0

        if T <= 0:
            await interaction.followup.send(f"❌ Expiration date must be in the future.")
            return

        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        current_price = float(hist["Close"][-1])

        hist["returns"] = np.log(hist["Close"] / hist["Close"].shift(1))
        hist_vol = np.std(hist["returns"].dropna()) * np.sqrt(252)

        iv = None
        try:
            options = stock.option_chain(expiration)
            chain = options.calls if option_type == "call" else options.puts
            match = chain[chain["strike"] == strike]
            if not match.empty and "impliedVolatility" in match.columns:
                iv_val = match["impliedVolatility"].values[0]
                iv = iv_val if iv_val > 0 else None
        except Exception as e:
            pass

        volatility = iv if iv else hist_vol
        r = 0.015

        d1 = (np.log(current_price / strike) + (r + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        if option_type == "call":
            prob_itm = norm.cdf(d2)
            pop = norm.cdf(d1)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (current_price * volatility * np.sqrt(T))
            theta = (-current_price * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) - r * strike * np.exp(-r * T) * norm.cdf(d2)) / 365
            vega = current_price * norm.pdf(d1) * np.sqrt(T) / 100
        else:
            prob_itm = norm.cdf(-d2)
            pop = norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (current_price * volatility * np.sqrt(T))
            theta = (-current_price * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) + r * strike * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = current_price * norm.pdf(d1) * np.sqrt(T) / 100

        embed = discord.Embed(
            title=f"{ticker} {option_type.capitalize()} Option @ ${strike} expiring {expiration}",
            description=f"📊 Based on Black-Scholes model with {'Implied' if iv else 'Historical'} Volatility",
            color=discord.Color.blue()
        )
        embed.add_field(name="Current Price", value=f"${current_price:.2f}", inline=True)
        embed.add_field(name="Volatility Used", value=f"{volatility*100:.2f}%", inline=True)
        embed.add_field(name="Probability ITM", value=f"{prob_itm*100:.2f}%", inline=True)
        embed.add_field(name="Estimated POP", value=f"{pop*100:.2f}%", inline=True)
        embed.add_field(name="Delta", value=f"{delta:.4f}", inline=True)
        embed.add_field(name="Gamma", value=f"{gamma:.4f}", inline=True)
        embed.add_field(name="Theta", value=f"{theta:.4f}", inline=True)
        embed.add_field(name="Vega", value=f"{vega:.4f}", inline=True)
        embed.set_footer(text="Estimates only | Not financial advice")
        embed.timestamp = discord.utils.utcnow()

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Error analyzing option:\n```{str(e)}```")

bot.run(os.getenv("DISCORD_TOKEN"))
