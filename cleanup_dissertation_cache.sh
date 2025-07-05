#!/bin/bash
# RAG Dissertation Cache Cleanup Script
# Removes models not needed for your RAG prompt engineering experiment

echo "🧹 RAG DISSERTATION CACHE CLEANUP"
echo "=================================="
echo "This will delete models NOT needed for your RAG experiment"
echo "Estimated space savings: ~15.7GB"
echo ""
echo "🔒 KEEPING (Essential for dissertation):"
echo "  ✅ facebook/rag-sequence-nq (2.0GB) - Main RAG model"
echo "  ✅ facebook/bart-large (1.9GB) - Generator"
echo "  ✅ facebook/dpr-question_encoder-single-nq-base (837MB)"
echo "  ✅ facebook/dpr-ctx_encoder-single-nq-base (837MB)"
echo "  ✅ sentence-transformers/all-mpnet-base-v2 (419MB)"
echo ""
echo "🗑️  DELETING (Not needed for RAG experiment):"
echo "  ❌ facebook/rag-token-nq (3.9GB) - Alternative RAG (not using)"
echo "  ❌ facebook/rag-token-base (3.9GB) - Base model (not using)"
echo "  ❌ facebook/rag-sequence-base (3.9GB) - Base model (not using)"
echo "  ❌ Qwen/Qwen1.5-1.8B-Chat (3.5GB) - Different LLM"
echo "  ❌ microsoft/DialoGPT-large (1.7GB) - Already have better"
echo "  ❌ Plus 10 smaller models (~2.7GB total)"
echo ""

read -p "❓ Proceed with cleanup? This CANNOT be undone! (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled - no changes made"
    exit 0
fi

echo ""
echo "🗑️  Starting cleanup process..."

# Large models to delete (3.5GB+)
echo "Deleting large alternative RAG models..."
rm -rf ~/.cache/huggingface/hub/models--facebook--rag-token-nq/
echo "  ✅ Deleted rag-token-nq (3.9GB)"

rm -rf ~/.cache/huggingface/hub/models--facebook--rag-token-base/
echo "  ✅ Deleted rag-token-base (3.9GB)"

rm -rf ~/.cache/huggingface/hub/models--facebook--rag-sequence-base/
echo "  ✅ Deleted rag-sequence-base (3.9GB)"

rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/
echo "  ✅ Deleted Qwen1.5-1.8B-Chat (3.5GB)"

# Medium models to delete (1GB+)
echo "Deleting medium-sized unused models..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--DialoGPT-large/
echo "  ✅ Deleted DialoGPT-large (1.7GB)"

rm -rf ~/.cache/huggingface/hub/models--facebook--bart-large-cnn/
echo "  ✅ Deleted bart-large-cnn (1.6GB)"

# Smaller models to delete
echo "Deleting smaller unused models..."
rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-base/
echo "  ✅ Deleted flan-t5-base (948MB)"

rm -rf ~/.cache/huggingface/hub/models--t5-base/
echo "  ✅ Deleted t5-base (853MB)"

rm -rf ~/.cache/huggingface/hub/models--allenai--unifiedqa-t5-base/
echo "  ✅ Deleted unifiedqa-t5-base (852MB)"

rm -rf ~/.cache/huggingface/hub/models--facebook--dpr-ctx_encoder-multiset-base/
echo "  ✅ Deleted dpr-ctx_encoder-multiset-base (837MB)"

rm -rf ~/.cache/huggingface/hub/models--deepset--roberta-base-squad2/
echo "  ✅ Deleted roberta-base-squad2 (475MB)"

rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-small/
echo "  ✅ Deleted flan-t5-small (297MB)"

rm -rf ~/.cache/huggingface/hub/models--google--t5-small-ssm-nq/
echo "  ✅ Deleted t5-small-ssm-nq (295MB)"

rm -rf ~/.cache/huggingface/hub/models--datarpit--distilbert-base-uncased-finetuned-natural-questions/
echo "  ✅ Deleted distilbert-natural-questions (255MB)"

rm -rf ~/.cache/huggingface/hub/models--distilbert-base-uncased-distilled-squad/
echo "  ✅ Deleted distilbert-squad (254MB)"

echo ""
echo "🎉 CLEANUP COMPLETED!"
echo ""

# Show current cache size
echo "📊 Current cache size:"
du -sh ~/.cache/huggingface/

echo ""
echo "🔍 Remaining RAG models:"
du -sh ~/.cache/huggingface/hub/models--facebook--rag-sequence-nq/
du -sh ~/.cache/huggingface/hub/models--facebook--bart-large/
du -sh ~/.cache/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/
du -sh ~/.cache/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/
du -sh ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/

echo ""
echo "✅ Your cache is now optimized for RAG dissertation!"
echo "💾 Space freed: ~15.7GB"
echo "🎯 Ready for production RAG experiment!"