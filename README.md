# Discord Stock + Option Bot

This bot gives real-time analysis of stocks and options using slash commands.

## Features

- `/stock` — VWAP, RSI(5), MACD(6,13,5)
- `/option` — Probability ITM, POP, Delta, Gamma, Theta, Vega
- Automatically uses Implied Volatility (IV) if available

## Run locally

```bash
pip install -r requirements.txt
python bot.py
```

## Deploy to Seenode

1. Push this folder to GitHub
2. Connect repo to Seenode
3. Add your `DISCORD_TOKEN` as an environment variable
4. Seenode will auto-start with the Procfile
