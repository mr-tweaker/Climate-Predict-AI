#!/usr/bin/env python3
"""
ClimatePredict AI - Main Application
Weather Prediction and Climate Analysis using AI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from models.weather_predictor import WeatherPredictor
from models.climate_analyzer import ClimateAnalyzer

def main():
    """Main application entry point"""
    logger = setup_logger()
    
    logger.info("🌍 Starting ClimatePredict AI")
    logger.info("=" * 50)
    
    # Initialize components
    weather_predictor = WeatherPredictor()
    climate_analyzer = ClimateAnalyzer()
    
    logger.info("✅ ClimatePredict AI initialized successfully!")
    
    # Add your main application logic here
    print("🚀 ClimatePredict AI is ready!")
    print("Run 'streamlit run src/web_app/app.py' to start the web interface")

if __name__ == "__main__":
    main()
