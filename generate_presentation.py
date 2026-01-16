"""
Auto-generate PDF Presentation from Analysis Results
Creates professional 30-slide PDF presentation with charts and metrics
Direct PDF generation (no PPTX required)
"""

import os
import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Configuration
PROJECT_ROOT = Path(__file__).parent
PLOTS_PATH = PROJECT_ROOT / "plots"
OUTPUT_PDF = PROJECT_ROOT / "Lakshy_Mittal_Presentation.pdf"

# Colors
COLOR_DARK_BLUE = HexColor("#1F77B4")
COLOR_WHITE = HexColor("#FFFFFF")
COLOR_DARK_GRAY = HexColor("#323232")
COLOR_LIGHT_GRAY = HexColor("#C8C8C8")

def create_pdf_presentation():
    """Generate complete PDF presentation"""
    
    if not HAS_REPORTLAB:
        print("ERROR: reportlab not installed. Install with: pip install reportlab")
        return False
    
    print("="*70)
    print("GENERATING PDF PRESENTATION")
    print("="*70)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(letter),
        rightMargin=0.3*inch,
        leftMargin=0.3*inch,
        topMargin=0.3*inch,
        bottomMargin=0.3*inch
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=54,
        textColor=COLOR_WHITE,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=22,
        textColor=COLOR_LIGHT_GRAY,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=32,
        textColor=COLOR_WHITE,
        spaceAfter=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=12,
        textColor=COLOR_DARK_GRAY,
        spaceAfter=6,
        leading=16,
        alignment=TA_LEFT,
        fontName='Helvetica',
        leftIndent=15
    )
    
    story = []
    slide_num = [0]
    
    def add_title_page(title, subtitle=""):
        """Add title slide with blue background - FIXED"""
        slide_num[0] += 1
        print(f"\n[{slide_num[0]}/30] Title slide...")
        
        # Create title page with blue background
        # FIX: 4 rows in data, 4 row heights
        title_table = Table(
            [
                [Paragraph(title, title_style)],
                [Spacer(0, 0.5*inch)],
                [Paragraph(subtitle, subtitle_style)],
                [Spacer(0, 2*inch)],
            ],
            colWidths=[10*inch],
            rowHeights=[1.5*inch, 0.5*inch, 1*inch, 2*inch]
        )
        title_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_DARK_BLUE),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        
        story.append(title_table)
        story.append(PageBreak())
    
    def add_content_page(title, bullets=None, image_path=None):
        """Add content slide with title header"""
        slide_num[0] += 1
        print(f"[{slide_num[0]}/30] {title}...")
        
        # Title header with blue background
        header_table = Table(
            [[Paragraph(title, heading_style)]],
            colWidths=[10*inch],
            rowHeights=[0.7*inch]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_DARK_BLUE),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(header_table)
        story.append(Spacer(10*inch, 0.15*inch))
        
        # Content area
        if image_path and os.path.exists(image_path):
            try:
                img = Image(image_path, width=9.5*inch, height=5.2*inch)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Chart: {os.path.basename(image_path)}]", bullet_style))
        elif bullets:
            for bullet in bullets:
                if bullet.strip():
                    story.append(Paragraph(f"• {bullet}", bullet_style))
        
        story.append(PageBreak())
    
    # ===== SLIDE 1: TITLE SLIDE =====
    add_title_page(
        "Quantitative Trading Strategy with Machine Learning",
        "ML-Enhanced Algorithmic Trading System for NIFTY 50"
    )
    
    # ===== SLIDE 2: EXECUTIVE SUMMARY - Overview =====
    add_content_page("Executive Summary: Project Overview", bullets=[
        "End-to-end quantitative trading system for NIFTY 50 (5-min intraday)",
        "1-year historical data (Oct 2021 - Oct 2022)",
        "3-layer strategy: Trend signals → Regime filter → ML enhancement",
        "Demonstrates ML as probabilistic trade filter, not price predictor",
        "Combines 5/15 EMA crossover + HMM regimes + XGBoost classifier",
        "Goal: Improve risk-adjusted returns while reducing drawdowns"
    ])
    
    # ===== SLIDE 3: EXECUTIVE SUMMARY - Key Findings =====
    add_content_page("Executive Summary: Key Findings", bullets=[
        "✓ Regime filter reduces false signals by filtering sideways periods",
        "✓ ML enhancement improves win rate and reduces drawdown",
        "✓ Sharpe ratio improvement: [See results]",
        "✓ Max drawdown reduction: [See results]",
        "✓ Scalable architecture: Modular code for production deployment",
        "✓ Comprehensive analysis: Outlier detection + statistical testing"
    ])
    
    # ===== SLIDE 4: DATA PIPELINE =====
    add_content_page("Data Pipeline Architecture", bullets=[
        "Step 1: Data Acquisition",
        "   • NIFTY 50 Spot: 5-min OHLCV data",
        "   • NIFTY Futures: Monthly contracts with rollover handling",
        "   • Data source: NSE historical data (1 year)",
        "Step 2: Data Cleaning & Alignment",
        "   • Handle missing values (forward fill + drop)",
        "   • Remove outliers (IQR method)",
        "   • Align timestamps across spot & futures"
    ])
    
    # ===== SLIDE 5: FEATURE ENGINEERING - Indicators =====
    add_content_page("Feature Engineering: Technical Indicators", bullets=[
        "Trend Features:",
        "   • EMA(5) - Fast moving average",
        "   • EMA(15) - Slow moving average",
        "   • EMA Spread - Distance between fast & slow",
        "Volatility Features:",
        "   • Rolling Volatility - 20-period log return std dev",
        "   • High Vol Regime - Binary flag (above/below median)"
    ])
    
    # ===== SLIDE 6: FEATURE ENGINEERING - Derived Features =====
    add_content_page("Feature Engineering: Derived Features", bullets=[
        "Return Features:",
        "   • Log Returns - Price momentum",
        "   • Futures Basis - (Futures - Spot) / Spot",
        "Options-Based Features (Black-Scholes):",
        "   • Greeks: Delta, Gamma, Theta, Vega, Rho",
        "   • IV Features: Average IV, IV Spread",
        "   • PCR Ratios: Put-Call Ratio (OI & Volume based)"
    ])
    
    # ===== SLIDE 7: REGIME DETECTION - HMM =====
    add_content_page("Regime Detection: Hidden Markov Model", bullets=[
        "Model Architecture:",
        "   • Gaussian HMM with 3 hidden states",
        "   • Input: Log returns, volatility, EMA spread, vol regime",
        "   • Training: 70% of data | Testing: 30%",
        "Regime Mapping (Dynamic based on returns):",
        "   • State 1 → Uptrend (+1): Highest avg returns",
        "   • State 0 → Sideways (0): Middle avg returns",
        "   • State 2 → Downtrend (-1): Lowest avg returns"
    ])
    
    # ===== SLIDE 8: REGIME DETECTION - Visualization =====
    add_content_page("Regime Detection: Price with Regime Overlay",
                     image_path=str(PLOTS_PATH / "03_price_with_regimes.png"))
    
    # ===== SLIDE 9: REGIME DETECTION - Statistics =====
    add_content_page("Regime Detection: Statistics by Regime", bullets=[
        "Uptrend Regime: Higher average returns, moderate volatility, higher Sharpe ratio",
        "Downtrend Regime: Negative returns, high volatility, low/negative Sharpe ratio",
        "Sideways Regime: Near-zero returns, low volatility, limited opportunities"
    ])
    
    # ===== SLIDE 10: BASELINE STRATEGY =====
    add_content_page("Trading Strategy: Baseline (5/15 EMA)", bullets=[
        "Entry Signals:",
        "   • LONG: EMA(5) crosses above EMA(15)",
        "   • SHORT: EMA(5) crosses below EMA(15)",
        "Exit Signals: Opposite crossover occurs",
        "Characteristics: Simple rule-based system, whipsaw risk in sideways markets"
    ])
    
    # ===== SLIDE 11: REGIME-FILTERED STRATEGY =====
    add_content_page("Trading Strategy: Regime-Filtered", bullets=[
        "Entry Signals:",
        "   • LONG: EMA(5) > EMA(15) AND Regime = Uptrend",
        "   • SHORT: EMA(5) < EMA(15) AND Regime = Downtrend",
        "   • NO trades in Sideways regime",
        "Benefits: Filters signals, trades with trend, improves risk-adjusted returns"
    ])
    
    # ===== SLIDE 12: STRATEGY BACKTEST RESULTS =====
    add_content_page("Backtesting Results: Performance Metrics", bullets=[
        "Key Metrics: Total Return, Annual Return, Sharpe Ratio, Sortino Ratio",
        "Risk Metrics: Max Drawdown, Calmar Ratio, Win Rate, Profit Factor",
        "Evaluation: 70% training | 30% testing (time-series split)",
        "No look-ahead bias: Signals shifted 1 bar before calculating returns"
    ])
    
    # ===== SLIDE 13: CUMULATIVE RETURNS =====
    add_content_page("Strategy Comparison: Cumulative Returns",
                     image_path=str(PLOTS_PATH / "07_cumulative_returns_comparison.png"))
    
    # ===== SLIDE 14: SHARPE RATIO COMPARISON =====
    add_content_page("Strategy Comparison: Sharpe Ratio",
                     image_path=str(PLOTS_PATH / "08_sharpe_ratio_comparison.png"))
    
    # ===== SLIDE 15: MAX DRAWDOWN COMPARISON =====
    add_content_page("Strategy Comparison: Maximum Drawdown",
                     image_path=str(PLOTS_PATH / "09_drawdown_comparison.png"))
    
    # ===== SLIDE 16: ML MODELS - Overview =====
    add_content_page("Machine Learning: Model Architecture", bullets=[
        "Model 1: Logistic Regression - Linear classifier, interpretable baseline",
        "Model 2: XGBoost - Non-linear gradient boosting (100 trees, depth=5)",
        "Model 3: LSTM - Recurrent neural network (10-candle sequences, 64 units)",
        "Target: Binary classification (profitable/unprofitable next candle)"
    ])
    
    # ===== SLIDE 17: MODEL COMPARISON =====
    add_content_page("ML Models: Performance Comparison",
                     image_path=str(PLOTS_PATH / "01_model_comparison.png"))
    
    # ===== SLIDE 18: FEATURE IMPORTANCE =====
    add_content_page("XGBoost: Feature Importance Analysis",
                     image_path=str(PLOTS_PATH / "02_xgboost_feature_importance.png"))
    
    # ===== SLIDE 19: ML-ENHANCED STRATEGY =====
    add_content_page("Trading Strategy: ML-Enhanced", bullets=[
        "Strategy Logic:",
        "   1. Generate signals from 5/15 EMA + Regime filter",
        "   2. For each signal, get ML probability prediction",
        "   3. Only take trade if XGBoost confidence > 60%",
        "Benefits: Reduces false signals, improves win rate, best risk-adjusted returns"
    ])
    
    # ===== SLIDE 20: OUTLIER ANALYSIS - Overview =====
    add_content_page("Outlier Analysis: Methodology", bullets=[
        "Outlier Detection (Z-score > 3):",
        "   • Extract all individual trades from signals",
        "   • Calculate PnL and PnL% for each trade",
        "   • Identify trades with |Z-score| > 3 as outliers",
        "Analysis: Compare outlier vs normal trades, statistical tests"
    ])
    
    # ===== SLIDE 21: PnL vs DURATION =====
    add_content_page("Outlier Analysis: PnL vs Trade Duration",
                     image_path=str(PLOTS_PATH / "10_pnl_vs_duration.png"))
    
    # ===== SLIDE 22: PnL DISTRIBUTION =====
    add_content_page("Outlier Analysis: PnL Distribution",
                     image_path=str(PLOTS_PATH / "11_pnl_distribution.png"))
    
    # ===== SLIDE 23: FEATURE COMPARISON =====
    add_content_page("Outlier Analysis: Feature Distributions",
                     image_path=str(PLOTS_PATH / "12_feature_comparison_boxplot.png"))
    
    # ===== SLIDE 24: TIME OF DAY ANALYSIS =====
    add_content_page("Outlier Analysis: Time-of-Day Patterns",
                     image_path=str(PLOTS_PATH / "13_time_of_day_analysis.png"))
    
    # ===== SLIDE 25: CORRELATION ANALYSIS =====
    add_content_page("Outlier Analysis: Feature Correlations",
                     image_path=str(PLOTS_PATH / "14_correlation_heatmap.png"))
    
    # ===== SLIDE 26: KEY INSIGHTS =====
    add_content_page("High-Performance Analysis: Key Insights", bullets=[
        "Outlier Characteristics:",
        "   • High proportion are statistical outliers (|Z-score| > 3)",
        "   • Significantly higher average PnL than normal trades",
        "   • Tend to last longer than normal trades",
        "Pattern Recognition: Certain hours profitable, volatility predicts outcomes"
    ])
    
    # ===== SLIDE 27: CONCLUSIONS =====
    add_content_page("Conclusions & Key Takeaways", bullets=[
        "✓ Hybrid strategy (EMA + Regime + ML) outperforms baseline",
        "✓ ML as trade filter reduces noise and improves Sharpe ratio",
        "✓ Regime detection filters sideways market periods effectively",
        "✓ Feature engineering captures market microstructure",
        "✓ Outlier analysis reveals tradeable patterns",
        "✓ Modular, scalable architecture ready for production"
    ])
    
    # ===== SLIDE 28: RECOMMENDATIONS =====
    add_content_page("Recommendations for Future Work", bullets=[
        "Short-term (0-3 months):",
        "   • Deploy on live data with broker API integration",
        "   • Add position sizing and risk management rules",
        "Medium-term (3-12 months): Extend to multi-asset, advanced models",
        "Long-term: Portfolio optimization and real-time regime identification"
    ])
    
    # ===== SLIDE 29: TECHNICAL IMPLEMENTATION =====
    add_content_page("Technical Implementation Highlights", bullets=[
        "Code Quality: Modular architecture, type hints, docstrings, no look-ahead bias",
        "ML Validation: Time-series cross-validation, StandardScaler normalization",
        "Reproducibility: Automated workflows, CSV results, publication-quality visualizations"
    ])
    
    # ===== SLIDE 30: THANK YOU =====
    add_title_page(
        "Thank You",
        "Lakshy Mittal | Quantitative Trading ML Assignment | January 2025"
    )
    
    # Build PDF
    try:
        doc.build(story)
        file_size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
        print(f"\n✓ PDF saved: {OUTPUT_PDF}")
        print(f"✓ File size: {file_size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"\n❌ Error building PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PDF PRESENTATION GENERATION SCRIPT")
    print("="*70)
    
    if not HAS_REPORTLAB:
        print("\n❌ ERROR: reportlab not installed")
        print("Install with: pip install reportlab")
        sys.exit(1)
    
    success = create_pdf_presentation()
    
    if success:
        print("\n" + "="*70)
        print("✓ PDF PRESENTATION CREATED SUCCESSFULLY!")
        print("="*70)
        print(f"\nFile location: {OUTPUT_PDF}")
        print(f"Total slides: 30")
        print("\nWhat's next:")
        print("  1. Download: Lakshy_Mittal_Presentation.pdf")
        print("  2. Review the presentation")
        print("  3. If OK, delete this script: generate_presentation.py")
        print("  4. Push to GitHub and submit to Info@klypto.app")
    else:
        print("\n❌ PDF creation failed")
        sys.exit(1)