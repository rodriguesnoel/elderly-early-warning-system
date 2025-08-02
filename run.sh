#!/bin/bash

# End-to-End ML Pipeline Execution Script

echo "🚀 Starting ML Pipeline..."

python src/data_loader.py
python src/preprocessing.py
python src/feature_engineer.py
python src/model_trainer.py
python src/model_evaluator.py

echo "✅ ML Pipeline completed successfully."
