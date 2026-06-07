"""Representative large-cap Taiwan stocks used by the dashboard."""

from __future__ import annotations


TAIWAN_STOCKS = [
    {"symbol": "2330.TW", "code": "2330", "name": "台積電", "sector": "半導體"},
    {"symbol": "2454.TW", "code": "2454", "name": "聯發科", "sector": "半導體"},
    {"symbol": "2308.TW", "code": "2308", "name": "台達電", "sector": "電子零組件"},
    {"symbol": "2317.TW", "code": "2317", "name": "鴻海", "sector": "電子製造"},
    {"symbol": "3711.TW", "code": "3711", "name": "日月光投控", "sector": "半導體"},
    {"symbol": "2303.TW", "code": "2303", "name": "聯電", "sector": "半導體"},
    {"symbol": "2383.TW", "code": "2383", "name": "台光電", "sector": "電子零組件"},
    {"symbol": "2345.TW", "code": "2345", "name": "智邦", "sector": "通訊網路"},
    {"symbol": "3037.TW", "code": "3037", "name": "欣興", "sector": "電子零組件"},
    {"symbol": "2327.TW", "code": "2327", "name": "國巨", "sector": "電子零組件"},
    {"symbol": "2891.TW", "code": "2891", "name": "中信金", "sector": "金融"},
    {"symbol": "2382.TW", "code": "2382", "name": "廣達", "sector": "電腦及週邊"},
    {"symbol": "2881.TW", "code": "2881", "name": "富邦金", "sector": "金融"},
    {"symbol": "2360.TW", "code": "2360", "name": "致茂", "sector": "電子儀器"},
    {"symbol": "2882.TW", "code": "2882", "name": "國泰金", "sector": "金融"},
    {"symbol": "2886.TW", "code": "2886", "name": "兆豐金", "sector": "金融"},
    {"symbol": "2884.TW", "code": "2884", "name": "玉山金", "sector": "金融"},
    {"symbol": "2885.TW", "code": "2885", "name": "元大金", "sector": "金融"},
    {"symbol": "2887.TW", "code": "2887", "name": "台新新光金", "sector": "金融"},
    {"symbol": "1216.TW", "code": "1216", "name": "統一", "sector": "食品"},
    {"symbol": "1301.TW", "code": "1301", "name": "台塑", "sector": "塑膠"},
    {"symbol": "1303.TW", "code": "1303", "name": "南亞", "sector": "塑膠"},
    {"symbol": "2002.TW", "code": "2002", "name": "中鋼", "sector": "鋼鐵"},
    {"symbol": "2412.TW", "code": "2412", "name": "中華電", "sector": "通訊網路"},
    {"symbol": "3008.TW", "code": "3008", "name": "大立光", "sector": "光電"},
    {"symbol": "2379.TW", "code": "2379", "name": "瑞昱", "sector": "半導體"},
    {"symbol": "2357.TW", "code": "2357", "name": "華碩", "sector": "電腦及週邊"},
    {"symbol": "3045.TW", "code": "3045", "name": "台灣大", "sector": "通訊網路"},
    {"symbol": "5871.TW", "code": "5871", "name": "中租-KY", "sector": "金融"},
    {"symbol": "6505.TW", "code": "6505", "name": "台塑化", "sector": "油電燃氣"},
]

DEFAULT_TICKERS = [stock["symbol"] for stock in TAIWAN_STOCKS[:10]]
VALID_TICKERS = {stock["symbol"] for stock in TAIWAN_STOCKS}
