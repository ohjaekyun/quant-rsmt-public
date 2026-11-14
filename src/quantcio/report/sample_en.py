"""English text of the deterministic SAMPLE report (report/sample.py `_SPECS`), used when
settings.report.language == "en".

Every entry mirrors one Korean spec field by field and item by item (same list lengths, same order, same numbers);
tests/test_report.py checks the parity. Korean amounts are written with English scale words of the same value
("10조 4,000억원" -> "KRW 10.4 trillion"). Synthetic text only: nothing here is real news or a real filing.
"""

from __future__ import annotations

from typing import Any

SAMPLE_TEXT_EN: dict[str, dict[str, Any]] = {
    "005930": {
        "name": "Samsung Electronics",
        "sector": "Semiconductors",
        "analyst_rating": "4.05 Buy",
        "news": {
            "key_positives": [
                "HBM4 12-high passes NVIDIA qualification; Q4 supply ramps",
                "Q3 operating profit seen back in the KRW 10 trillion range",
            ],
            "key_negatives": ["Foundry 2nm yields remain weak"],
            "key_events": ["Additional KRW 3 trillion share buyback approved"],
        },
        "filings": {
            "summary": [
                "Q3 preliminary operating profit KRW 10.4 trillion, up 145.2% YoY",
                "Higher memory prices and a larger HBM mix drive the profit recovery",
                "Extra KRW 3 trillion buyback expands shareholder returns",
            ],
            "earnings_highlights": [
                "Revenue KRW 79.1 trillion (+12.3% YoY)",
                "Operating profit KRW 10.4 trillion (+145.2% YoY)",
            ],
            "risk_factors": [
                (
                    "Tighter US chip export controls",
                    "Reports of further curbs on equipment and HBM exports to China",
                    "News: concerns over tighter US chip export controls",
                ),
                (
                    "Foundry yield / order delays",
                    "Weak 2nm yields may delay large customer orders",
                    "News: weak foundry yields",
                ),
                ("Union strike", "Wage talks collapsed; strike announced", "News: union strike notice"),
            ],
            "sources": ["Consolidated preliminary earnings (fair disclosure)", "Share buyback decision"],
        },
        "thesis": "HBM4 ramp and a memory price rebound give strong earnings momentum, and quant and text signals "
        "agree. Scale in to manage short-term volatility from export-control headlines.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "AAPL": {
        "name": "Apple",
        "sector": "Technology",
        "analyst_rating": "Buy",
        "news": {
            "key_positives": ["iPhone 18 early sales ahead of the prior model"],
            "key_negatives": [
                "Smartphone share in China declining",
                "App Store antitrust appeal ruling imminent",
            ],
            "key_events": ["Staged rollout of AI features"],
        },
        "filings": {
            "summary": [
                "Record services revenue; gross margin up to 47.1%",
                "Tariff-related costs of $1.1 billion booked in the quarter",
                "$25 billion of buybacks executed; dividend raised 4%",
            ],
            "earnings_highlights": [
                "Revenue $94.9 billion (+6% YoY)",
                "EPS $1.64 (+12% YoY)",
            ],
            "risk_factors": [
                (
                    "App Store antitrust suit",
                    "Losing the appeal would force a change to the fee structure",
                    "10-Q Item 1A",
                ),
                ("China demand slowdown", "Local brands erode Greater China revenue", "10-Q MD&A"),
                ("Tariff costs", "China tariffs keep weighing on costs", "8-K EX-99.1"),
            ],
            "sources": ["10-Q (2026-08-01)", "8-K Item 2.02 EX-99.1"],
        },
        "thesis": "Top-tier quality, but valuation and an unskilled ML model leave little directional edge. "
        "Reassess once volatility after the antitrust ruling is clear.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": ("critic: server fallback path used (see usage.iterations)",),
    },
    "000660": {
        "name": "SK hynix",
        "sector": "Semiconductors",
        "analyst_rating": "4.20 Buy",
        "news": {
            "key_positives": [
                "HBM4 volume shipments begin; share seen above 50%",
                "Q3 operating profit expected to beat consensus",
            ],
            "key_negatives": ["Debate over a memory-cycle peak resurfaces"],
            "key_events": ["Cheongju M15X fab starts early"],
        },
        "filings": {
            "summary": [
                "Q2 operating profit KRW 9.2 trillion; record 46% margin",
                "HBM passes 40% of revenue, improving the product mix",
                "Larger M15X fab investment lifts the capex plan",
            ],
            "earnings_highlights": [
                "Revenue KRW 20.1 trillion (+58% YoY)",
                "Operating profit KRW 9.2 trillion (+92% YoY)",
            ],
            "risk_factors": [
                (
                    "Memory-cycle peak risk",
                    "Commodity DRAM price gains may slow in H2",
                    "Broker research summary",
                ),
                (
                    "Customer concentration",
                    "A large share of HBM revenue comes from one customer",
                    "Half-year report",
                ),
                ("Capex burden", "New fab spending makes free cash flow more volatile", "IR materials"),
            ],
            "sources": ["Preliminary earnings (fair disclosure)", "Half-year report"],
        },
        "thesis": "HBM leadership and accelerating earnings are confirmed; momentum, quality and text signals are "
        "all strong. Take 50% profit at the take-profit level given peak-cycle debate.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": ("news: 2 articles whose evidence quote did not match the source excluded from the score",),
    },
    "NVDA": {
        "name": "NVIDIA",
        "sector": "Semiconductors",
        "analyst_rating": "Strong Buy",
        "news": {
            "key_positives": [
                "Early shipment of next-gen Rubin GPUs confirmed",
                "Data-center revenue guidance raised",
            ],
            "key_negatives": [
                "Limits on H20 sales to China persist",
                "Customers' in-house AI chips add competition",
            ],
            "key_events": ["Shares up 3% after hours on results"],
        },
        "filings": {
            "summary": [
                "Data-center revenue up 56% YoY, 88% of total",
                "Next-quarter revenue guidance above expectations",
                "Guidance assumes no China revenue",
            ],
            "earnings_highlights": [
                "Revenue $46.7 billion (+56% YoY)",
                "Data Center revenue $41.1 billion",
            ],
            "risk_factors": [
                ("Export controls", "Advanced GPU sales to China remain restricted", "10-Q Item 1A"),
                ("Customer concentration", "Top customers' revenue share rising", "10-Q MD&A"),
                ("Supply constraints", "CoWoS packaging capacity constraints", "8-K EX-99.1"),
            ],
            "sources": ["8-K Item 2.02 EX-99.1", "10-Q (2026-08-27)"],
        },
        "thesis": "Strong results and guidance; quant and text signals both point to Buy. Keep the weight within "
        "the Kelly cap given high volatility and export-control headlines.",
        "contradictions": ("ML points up but 7-day news sentiment is trending down",),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "247540": {
        "name": "EcoPro BM",
        "sector": "Battery materials",
        "analyst_rating": "2.80 Neutral",
        "news": {
            "key_positives": ["North American cathode plant breaks ground"],
            "key_negatives": [
                "Lower lithium prices widen inventory write-downs",
                "Weaker EV sales at key customers cut shipments",
            ],
            "key_events": ["Denies rumor of a rights offering"],
        },
        "filings": {
            "summary": [
                "Q2 operating loss KRW 31.2 billion; losses continue",
                "Cathode volumes down 14% QoQ",
                "Inventory write-downs hurt profitability",
            ],
            "earnings_highlights": [
                "Revenue KRW 642 billion (-28.1% YoY)",
                "Operating loss KRW 31.2 billion (loss continued)",
            ],
            "risk_factors": [
                (
                    "Larger inventory write-downs",
                    "Further losses if lithium prices stay weak",
                    "Preliminary earnings (fair disclosure)",
                ),
                ("Weaker customer demand", "Slower EV sales reduce shipments", "News: lower shipments"),
                (
                    "Weaker balance sheet",
                    "Continued losses may raise borrowing",
                    "Half-year report",
                ),
            ],
            "sources": ["Preliminary earnings (fair disclosure)", "Half-year report"],
        },
        "thesis": "Continued losses and weaker customer demand leave fundamentals, trend and text signals all weak. "
        "Confidence cut for RSMT degraded mode (offset input missing).",
        "contradictions": (),
        "llm_errors": (),
        "warnings": ("Valuation history fetch failed: HTTPError: 503 Service Unavailable",),
        "degraded": ("offset input missing (z)",),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "MSFT": {
        "name": "Microsoft",
        "sector": "Software",
        "analyst_rating": "Strong Buy",
        "news": {
            "key_positives": ["Azure growth re-accelerates (+39%)", "Paid Copilot seats growing"],
            "key_negatives": ["Rising AI infrastructure capex"],
            "key_events": ["OpenAI deal structure reworked"],
        },
        "filings": {
            "summary": [
                "Cloud revenue incl. Azure up 39%, above expectations",
                "Quarterly capex rises to $30 billion; heavier depreciation",
                "Operating margin held at 45%",
            ],
            "earnings_highlights": [
                "Revenue $76.4 billion (+18% YoY)",
                "Operating income $34.3 billion",
            ],
            "risk_factors": [
                ("Capex burden", "AI data-center spending slows FCF", "10-K MD&A"),
                ("Regulation", "EU cloud antitrust probe", "10-K Item 1A"),
            ],
            "sources": ["10-K (2026-07-30)", "8-K Item 2.02 EX-99.1"],
        },
        "thesis": "Cloud re-acceleration and steady margins make it a low-volatility Buy candidate. The capex "
        "impact on margins needs confirming next quarter.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "035420": {
        "name": "NAVER",
        "sector": "Internet",
        "analyst_rating": "4.00 Buy",
        "news": {
            "key_positives": ["HyperCLOVA X wins more enterprise clients"],
            "key_negatives": ["Commerce competition slows GMV growth"],
            "key_events": [],
        },
        "filings": None,
        "thesis": "AI service upside is offset by slower commerce growth, leaving a neutral signal. Confidence "
        "lowered because the filing summary is missing.",
        "contradictions": (),
        "llm_errors": ("filings: stop_reason=max_tokens — filing summary missing",),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "TSLA": {
        "name": "Tesla",
        "sector": "Automobiles",
        "analyst_rating": "Hold",
        "news": {
            "key_positives": ["Robotaxi service area expands"],
            "key_negatives": [
                "Q3 deliveries expected below consensus",
                "CEO-related uncertainty resurfaces",
            ],
            "key_events": ["Shares down 20%+ over one month"],
        },
        "filings": {
            "summary": [
                "Auto revenue down YoY; energy storage revenue growing",
                "Lower average selling prices cut auto gross margin",
                "Robotaxi and humanoid investment expanding",
            ],
            "earnings_highlights": [
                "Revenue $22.5 billion (-12% YoY)",
                "Energy storage deployments 9.6 GWh",
            ],
            "risk_factors": [
                ("Demand slowdown", "EV price competition cuts volume and margin", "10-Q MD&A"),
                ("Key-person risk", "CEO dependence and governance debate", "10-Q Item 1A"),
                ("Regulation / recalls", "NHTSA probe into autonomous driving", "8-K"),
            ],
            "sources": ["10-Q (2026-07-24)", "8-K"],
        },
        "thesis": "A sharp drop and broken trend keep factor and text signals bearish; the ML rebound call is "
        "unskilled and down-weighted. Trim until deliveries are confirmed.",
        "contradictions": ("The ML signal predicts a rebound (+1.5%) but news sentiment is worsening",),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": ("Hold off new entries until Q3 deliveries (10/2) are confirmed",),
        "llm_notices": (),
    },
    "051910": {
        "name": "LG Chem",
        "sector": "Chemicals",
        "analyst_rating": "3.60 Buy",
        "news": {
            "key_positives": ["Advanced materials unit turns profitable"],
            "key_negatives": [
                "Prolonged petrochemical downturn",
                "Lower value of its LG Energy Solution stake",
            ],
            "key_events": ["Considering sale of NCC assets"],
        },
        "filings": {
            "summary": [
                "Petrochemical unit still loss-making",
                "Advanced materials sales narrow the loss",
                "Possible dividend cut mentioned",
            ],
            "earnings_highlights": ["Operating profit KRW 121 billion (-72.5% YoY)"],
            "risk_factors": [
                (
                    "Petrochemical oversupply",
                    "Chinese capacity additions keep spreads weak",
                    "Half-year report",
                ),
                (
                    "Subsidiary value decline",
                    "LG Energy Solution share drop erodes stake value",
                    "News",
                ),
                ("Financial burden", "Rising borrowings", "Half-year report"),
            ],
            "sources": ["Half-year report", "Preliminary earnings (fair disclosure)"],
        },
        "thesis": "A weak petrochemical cycle and a sharp earnings drop keep quant and text signals bearish. "
        "Trim into rebounds.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "AMZN": {
        "name": "Amazon",
        "sector": "Consumer / Cloud",
        "analyst_rating": "Strong Buy",
        "news": {
            "key_positives": ["AWS growth back in the 20% range", "Advertising revenue keeps growing fast"],
            "key_negatives": ["FTC antitrust trial schedule set"],
            "key_events": [],
        },
        "filings": {
            "summary": [
                "AWS operating margin improves to 37%",
                "Record North America retail margin",
                "Record Prime Day sales",
            ],
            "earnings_highlights": [
                "Net sales $167.7 billion (+11% YoY)",
                "AWS segment sales $30.9 billion (+19% YoY)",
            ],
            "risk_factors": [
                (
                    "Antitrust suit",
                    "FTC case could force structural changes",
                    "10-Q Item 1A",
                ),
                ("Capex burden", "Rising AI infrastructure spending", "10-Q MD&A"),
            ],
            "sources": ["10-Q (2026-08-01)"],
        },
        "thesis": "AWS re-acceleration and better retail margins are lifting estimates. The antitrust case is a "
        "long-term variable with limited short-term impact on the signal.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "207940": {
        "name": "Samsung Biologics",
        "sector": "Biotech",
        "analyst_rating": "4.00 Buy",
        "news": {
            "key_positives": ["CMO contract signed with a large US pharma"],
            "key_negatives": ["Uncertainty over the US BIOSECURE Act"],
            "key_events": [],
        },
        "filings": {
            "summary": [
                "Q2 revenue KRW 1.2 trillion; Plant 4 utilization rising",
                "Plant 5 expansion on track",
                "Business restructuring after the spin-off",
            ],
            "earnings_highlights": [
                "Revenue KRW 1.2 trillion (+18% YoY)",
                "Operating profit KRW 400 billion (+21% YoY)",
            ],
            "risk_factors": [
                (
                    "Valuation burden",
                    "PER above 50x already prices in expectations",
                    "Broker research summary",
                ),
                (
                    "Policy uncertainty",
                    "US BIOSECURE Act rules not finalized",
                    "News",
                ),
            ],
            "sources": ["Half-year report", "Single sales/supply contract"],
        },
        "thesis": "Steady orders and growth, but valuation keeps the composite neutral. Reassess on new order "
        "disclosures.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": ("Volatility: EWMA(λ=0.94) fallback, 24.6% a year — GARCH not used: no convergence (flag=1)",),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
    "JPM": {
        "name": "JPMorgan Chase",
        "sector": "Banks",
        "analyst_rating": "Buy",
        "news": {
            "key_positives": [
                "Investment-banking fees up YoY",
                "Net interest income guidance raised",
            ],
            "key_negatives": ["Commercial real-estate credit concerns"],
            "key_events": [],
        },
        "filings": {
            "summary": [
                "Net interest income guidance raised",
                "Investment-banking fees recovering",
                "CET1 ratio 15.3%; ample capital",
            ],
            "earnings_highlights": ["Net revenue $45.7 billion (+8% YoY)", "EPS $5.24"],
            "risk_factors": [
                ("Credit risk", "Possible commercial real-estate loan losses", "10-Q"),
                ("Regulatory capital", "Basel III endgame would raise capital needs", "10-Q Item 1A"),
            ],
            "sources": ["10-Q (2026-08-05)"],
        },
        "thesis": "Stable earnings, ample capital and low volatility make it a defensive Buy candidate. Watch how "
        "the rate path affects net interest income.",
        "contradictions": (),
        "llm_errors": (),
        "warnings": (),
        "degraded": (),
        "extra_triggers": (),
        "llm_notices": (),
    },
}
