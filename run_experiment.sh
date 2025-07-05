#!/bin/bash
# Official RAG Experiment Runner
# Save as: run_experiment.sh

echo "🎯 OFFICIAL RAG PRODUCTION EXPERIMENT RUNNER"
echo "=============================================="
echo ""

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1  # Use both A100s
export TOKENIZERS_PARALLELISM=false  # Avoid warnings

# Create directory structure if needed
mkdir -p {scripts,data,results,logs,configs}

# Check if we're in the right directory
if [ ! -f "configs/config.yaml" ]; then
    echo "❌ Error: config.yaml not found in configs/"
    echo "Please ensure you're in the production_rag_experiment directory"
    exit 1
fi

echo "📁 Working directory: $(pwd)"
echo "🔧 Environment setup complete"
echo ""

# Step 1: System Check
echo "🔍 STEP 1: SYSTEM CHECK"
echo "----------------------"
cd scripts
python system_check.py

if [ $? -ne 0 ]; then
    echo "❌ System check failed. Please address issues before proceeding."
    exit 1
fi

echo ""
echo "✅ System check passed!"
echo ""

# Step 2: Confirm execution
echo "🚨 IMPORTANT NOTICE:"
echo "This experiment will download ~21GB of Wikipedia embeddings"
echo "This is a ONE-TIME download that will be cached for future use"
echo ""
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏹️  Experiment cancelled by user"
    exit 0
fi

# Step 3: Run the experiment
echo ""
echo "🚀 STEP 2: RUNNING PRODUCTION EXPERIMENT"
echo "----------------------------------------"
echo "⏰ This may take 30-60 minutes (mostly for download)"
echo "💡 Progress will be logged to logs/ directory"
echo ""

# Start the experiment
python production_rag_experiment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results saved to:"
    echo "   📁 results/detailed_results_*.csv"
    echo "   📁 results/summary_*.csv" 
    echo "   📁 results/report_*.md"
    echo ""
    echo "📝 Logs saved to:"
    echo "   📁 logs/system_check_*.log"
    echo "   📁 logs/official_rag_*.log"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review the markdown report for key findings"
    echo "   2. Analyze detailed CSV results"
    echo "   3. Use results in your dissertation"
    echo ""
else
    echo ""
    echo "❌ EXPERIMENT FAILED"
    echo "📝 Check logs/ directory for error details"
    echo "💡 You can re-run this script - downloads will resume"
    echo ""
fi

cd ..