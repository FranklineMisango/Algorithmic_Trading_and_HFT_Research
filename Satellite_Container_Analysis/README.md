# Satellite Container Analysis for Trading Signals

This project implements a satellite image analysis system to detect shipping containers at major ports and generate trading signals based on container volume changes, following the research methodology described in quantitative finance literature.

## Overview

The system:
1. Collects satellite imagery from major global ports using ESA Copernicus/Sentinel-2 data
2. Uses deep learning (YOLO) to detect and count shipping containers
3. Generates trading signals from container volume trends
4. Provides backtesting capabilities for signal validation

## Key Components

- `satellite_analysis.ipynb` - Main Jupyter notebook with complete workflow
- `data_collector.py` - ESA Copernicus API integration for satellite data
- `container_detector.py` - YOLO-based container detection
- `signal_generator.py` - Trading signal generation and backtesting
- `config.json` - API credentials and port coordinates

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure API credentials in `config.json`
2. Run the Jupyter notebook `satellite_analysis.ipynb`
3. The system will:
   - Search for satellite imagery of major ports
   - Detect containers using computer vision
   - Generate trading signals based on volume changes
   - Provide visualization and backtesting results

## Trading Signals

- **Bullish Signal**: Increasing container volumes, short MA > long MA
- **Bearish Signal**: Decreasing container volumes, short MA < long MA  
- **Warning Signal**: Extremely high volumes (potential bottleneck indicator)
- **Global Signal**: Aggregated signal across all monitored ports

## Major Ports Monitored

- Shanghai (China)
- Singapore
- Rotterdam (Netherlands) 
- Los Angeles (USA)
- Hamburg (Germany)

## Research Basis

Based on academic research showing strong correlation between satellite-observed container volumes and stock market performance across 27 countries, with average annual returns exceeding 16%.

## Data Sources

- ESA Copernicus Sentinel-2 satellite imagery
- Optional: Google Earth Engine integration
- Real-time container detection using YOLOv8

## Performance

The system aims to replicate research findings showing:
- Predictive power for global equity markets
- Enhanced performance during supply chain disruptions
- Early warning signals for economic slowdowns