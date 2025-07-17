#!/usr/bin/env python3
"""
7-Template Strategic RAG Experiment
Combines successful templates from 20-50 QA tests with strategic prompt patterns
Save as: seven_template_experiment.py
"""

import torch
import yaml
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SevenTemplateExperiment:
    def __init__(self, config_path="../configs/config3.yaml"):
        """Initialize 7-template experiment"""
        self.load_config(config_path)
        self.setup_logging()
        self.results = []

    def load_config(self, config_path):
        """Load experiment configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(" Configuration loaded successfully")
        except Exception as e:
            print(f" Failed to load config: {e}")
            # Use default config if file missing
            self.config = {
                'models': {'rag_model': 'facebook/rag-token-nq'},
                'experiment_settings': {'num_test_questions': 500},
                'output': {'results_directory': 'results', 'save_detailed_results': True}
            }
            print(" Using default configuration")

    def setup_logging(self):
        """Setup experiment logging"""
        self.experiment_id = f"seven_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = f"../logs/{self.experiment_id}.log"
        os.makedirs("../logs", exist_ok=True)
        print(f" Experiment ID: {self.experiment_id}")

    def log_message(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")

    def setup_gpu(self):
        """Setup GPU configuration"""
        self.log_message(" Setting up GPU configuration...")
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.log_message(" CUDA not available - using CPU")
            return False

        best_gpu = 0
        max_free = 0
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(
                i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free_mem = total_mem - cached
            if free_mem > max_free:
                max_free = free_mem
                best_gpu = i

        torch.cuda.set_device(best_gpu)
        self.device = f"cuda:{best_gpu}"
        self.log_message(f" Using device: {self.device}")
        return True

    def load_rag_models(self):
        """Load RAG models"""
        self.log_message(" Loading Facebook RAG models...")
        try:
            from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

            model_name = self.config['models']['rag_model']
            self.tokenizer = RagTokenizer.from_pretrained(model_name)

            self.retriever = RagRetriever.from_pretrained(
                model_name, index_name="exact", use_dummy_dataset=False)

            self.model = RagTokenForGeneration.from_pretrained(
                model_name, retriever=self.retriever).to(self.device)

            self.log_message(" RAG system ready!")
            return True
        except Exception as e:
            self.log_message(f" Failed to load RAG models: {e}")
            return False

    def load_squad_dataset(self):
        """Load SQuAD dataset and process questions"""
        self.log_message(" Loading SQuAD dataset for 7-template testing...")

        try:
            from datasets import load_dataset

            target_questions = self.config['experiment_settings']['num_test_questions']
            # Load extra to ensure we get enough
            load_size = min(target_questions + 500, 3000)

            self.log_message(f" Loading {load_size} SQuAD questions...")
            dataset = load_dataset(
                "squad_v2", split=f"validation[:{load_size}]")

            processed_questions = []

            for item in dataset:
                if item['answers']['text'] and len(item['answers']['text']) > 0:
                    question = item['question']
                    answer = item['answers']['text'][0]

                    # Keep answers under 12 words for better evaluation
                    if len(answer.split()) > 12:
                        continue

                    # Remove question mark for consistent baseline
                    base_question = question.rstrip('?').strip()

                    processed_questions.append({
                        'question': base_question,
                        'answer': answer,
                        'context': item['context'][:150] + "...",
                        'question_id': len(processed_questions)
                    })

                    if len(processed_questions) >= target_questions:
                        break

            self.test_questions = processed_questions

            self.log_message(
                f" Dataset loaded: {len(self.test_questions)} questions")
            self.log_message(
                f" Will test 7 templates = {len(self.test_questions) * 7} total evaluations")

            # Show examples
            self.log_message(f"\n SAMPLE QUESTIONS:")
            for i in range(min(3, len(self.test_questions))):
                q = self.test_questions[i]
                self.log_message(f"   {i+1}. Q: '{q['question']}'")
                self.log_message(f"      A: '{q['answer']}'")

            return True

        except Exception as e:
            self.log_message(f" Failed to load SQuAD dataset: {e}")
            return False

    def get_templates(self):
        """Get templates from config file"""
        try:
            templates = self.config.get('templates', [])
            if not templates:
                self.log_message(" No templates found in config!")
                return []

            self.log_message(
                f" Loaded {len(templates)} templates from config")
            return templates
        except Exception as e:
            self.log_message(f" Error loading templates from config: {e}")
            return []

    def generate_answer(self, question_text, template_info):
        """Generate answer using template from config"""
        try:
            # Format question according to template from config
            formatted_prompt = template_info["template"].format(
                question=question_text)

            input_dict = self.tokenizer.prepare_seq2seq_batch(
                formatted_prompt, return_tensors="pt")
            input_ids = input_dict["input_ids"].to(self.device)

            with torch.no_grad():
                generated = self.model.generate(input_ids=input_ids)

            answer = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True)[0]
            return answer.strip(), formatted_prompt

        except Exception as e:
            self.log_message(
                f"Generation error for {template_info['name']}: {e}")
            return f"Error: {str(e)[:30]}", formatted_prompt

    def calculate_metrics(self, generated, true_answer):
        """Calculate evaluation metrics"""
        gen_clean = generated.lower().strip()
        true_clean = true_answer.lower().strip()

        # Exact Match
        exact_match = 1.0 if gen_clean == true_clean else 0.0

        # F1 Score
        gen_tokens = gen_clean.split() if gen_clean else []
        true_tokens = true_clean.split() if true_clean else []

        gen_set = set(gen_tokens)
        true_set = set(true_tokens)

        if not gen_set and not true_set:
            f1_score = 1.0
        elif not gen_set or not true_set:
            f1_score = 0.0
        else:
            overlap = len(gen_set.intersection(true_set))
            precision = overlap / len(gen_set) if gen_set else 0
            recall = overlap / len(true_set) if true_set else 0

            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

        # Token overlap
        token_overlap = len(gen_set.intersection(true_set)) / \
            len(true_set) if true_set else 0.0

        return {
            'exact_match': exact_match,
            'f1_score': f1_score,
            'token_overlap': token_overlap
        }

    def run_experiment(self):
        """Run the 7-template experiment"""
        self.log_message(" Starting 7-template strategic experiment...")

        templates = self.get_templates()

        # Put basic template last to reduce order effects
        basic_template = next(t for t in templates if t["name"] == "basic")
        other_templates = [t for t in templates if t["name"] != "basic"]
        ordered_templates = other_templates + [basic_template]

        total_evaluations = len(self.test_questions) * len(templates)
        self.log_message(f" Experiment scope:")
        self.log_message(f"   Questions: {len(self.test_questions)}")
        self.log_message(f"   Templates: {len(templates)}")
        self.log_message(f"   Total evaluations: {total_evaluations}")

        # Show template details
        self.log_message(f"\n TEMPLATES FROM CONFIG:")
        for i, template in enumerate(ordered_templates, 1):
            self.log_message(
                f"   {i}. {template['name']}: \"{template['template']}\"")
            self.log_message(
                f"      Purpose: {template.get('purpose', 'Not specified')}")

        completed = 0

        for template in ordered_templates:
            template_name = template["name"]
            self.log_message(f"\n Testing template: {template_name}")

            template_results = []

            for i, question_data in enumerate(self.test_questions):
                question = question_data['question']
                true_answer = question_data['answer']

                try:
                    # Generate answer
                    generated_answer, formatted_prompt = self.generate_answer(
                        question, template)

                    # Calculate metrics
                    metrics = self.calculate_metrics(
                        generated_answer, true_answer)

                    # Store result
                    result = {
                        'experiment_id': self.experiment_id,
                        'template': template_name,
                        'template_purpose': template.get('purpose', 'Not specified'),
                        'question_id': i,
                        'question': question,
                        'formatted_prompt': formatted_prompt,
                        'true_answer': true_answer,
                        'generated_answer': generated_answer,
                        'exact_match': metrics['exact_match'],
                        'f1_score': metrics['f1_score'],
                        'token_overlap': metrics['token_overlap'],
                        'answer_length': len(generated_answer.split()),
                        'timestamp': datetime.now().isoformat()
                    }

                    self.results.append(result)
                    template_results.append(result)
                    completed += 1

                    # Progress update every 25 questions
                    if (i + 1) % 25 == 0:
                        progress = (completed / total_evaluations) * 100
                        recent_f1 = np.mean([r['f1_score']
                                            for r in template_results[-25:]])
                        self.log_message(
                            f"  Progress: {i+1}/{len(self.test_questions)} - Recent F1: {recent_f1:.3f}")

                except Exception as e:
                    self.log_message(
                        f"   Question {i+1} failed: {str(e)[:50]}")
                    continue

            # Template summary
            if template_results:
                avg_f1 = np.mean([r['f1_score'] for r in template_results])
                avg_em = np.mean([r['exact_match'] for r in template_results])
                self.log_message(
                    f"    {template_name}: F1={avg_f1:.3f}, EM={avg_em:.3f}")

        self.log_message(f"\n Experiment completed!")
        self.log_message(f" Total results generated: {len(self.results)}")

    def analyze_results(self):
        """Analyze 7-template results"""
        self.log_message("\n ANALYZING 7-TEMPLATE RESULTS...")

        if not self.results:
            self.log_message(" No results to analyze")
            return {}

        df = pd.DataFrame(self.results)

        # Calculate template metrics
        template_metrics = {}
        for template in df['template'].unique():
            template_data = df[df['template'] == template]

            template_metrics[template] = {
                'f1_score': template_data['f1_score'].mean(),
                'f1_std': template_data['f1_score'].std(),
                'exact_match': template_data['exact_match'].mean(),
                'token_overlap': template_data['token_overlap'].mean(),
                'avg_length': template_data['answer_length'].mean(),
                'num_samples': len(template_data),
                'purpose': template_data['template_purpose'].iloc[0]
            }

        # Display results
        self.log_message("\n 7-TEMPLATE PERFORMANCE RANKING:")
        self.log_message("-" * 85)
        self.log_message(
            f"{'Template':<20} {'F1 Score':<10} {'EM Score':<10} {'Samples':<8} {'Improvement':<12}")
        self.log_message("-" * 85)

        # Sort by F1 score
        sorted_templates = sorted(template_metrics.items(
        ), key=lambda x: x[1]['f1_score'], reverse=True)
        basic_f1 = template_metrics.get('basic', {}).get('f1_score', 0)

        for rank, (template, metrics) in enumerate(sorted_templates, 1):
            if template == 'basic':
                improvement = "BASELINE"
            else:
                if basic_f1 > 0:
                    pct = ((metrics['f1_score'] - basic_f1) / basic_f1 * 100)
                    improvement = f"{pct:+.1f}%"
                else:
                    improvement = "N/A"

            self.log_message(
                f"{template:<20} {metrics['f1_score']:.3f}     "
                f"{metrics['exact_match']:.3f}     "
                f"{metrics['num_samples']:<8} {improvement:<12}"
            )

        # Success analysis
        best_template = sorted_templates[0]
        basic_rank = next((i for i, (name, _) in enumerate(
            sorted_templates) if name == 'basic'), -1)

        self.log_message(f"\n EXPERIMENT SUCCESS ANALYSIS:")
        self.log_message(
            f"   Best Template: {best_template[0]} (F1: {best_template[1]['f1_score']:.3f})")
        self.log_message(f"     Purpose: {best_template[1]['purpose']}")

        if basic_rank == 0:
            self.log_message(f"   Basic template still dominates (rank 1)")
            self.log_message(f"   RAG may inherently prefer minimal prompts")
        else:
            self.log_message(
                f"   SUCCESS! Basic template relegated to rank {basic_rank + 1}")
            self.log_message(
                f"   Template engineering shows measurable impact!")

            # Show top performers vs basic
            self.log_message(f"\n TOP PERFORMERS VS BASIC:")
            for i, (template, metrics) in enumerate(sorted_templates[:3]):
                if template != 'basic':
                    improvement = (
                        (metrics['f1_score'] - basic_f1) / basic_f1 * 100) if basic_f1 > 0 else 0
                    self.log_message(
                        f"   {i+1}. {template}: {improvement:+.1f}% improvement")

        # Template category analysis
        self.log_message(f"\n TEMPLATE CATEGORY ANALYSIS:")
        instructional_templates = [
            'precise_instruction', 'direct_answer', 'generate_format', 'quoted_question']
        rag_specific = ['retrieval_focused']
        formatting = ['question_mark']

        for category, templates in [
            ("Instructional", instructional_templates),
            ("RAG-Specific", rag_specific),
            ("Formatting", formatting)
        ]:
            category_f1s = [template_metrics[t]['f1_score']
                            for t in templates if t in template_metrics]
            if category_f1s:
                avg_f1 = np.mean(category_f1s)
                improvement = ((avg_f1 - basic_f1) / basic_f1 *
                               100) if basic_f1 > 0 else 0
                self.log_message(
                    f"   {category}: Average F1={avg_f1:.3f} ({improvement:+.1f}% vs basic)")

        return template_metrics

    def save_results(self, template_metrics):
        """Save comprehensive results"""
        self.log_message("\n Saving results...")

        results_dir = f"../{self.config['output']['results_directory']}"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = f"{results_dir}/seven_template_detailed_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(detailed_file, index=False)
        self.log_message(f"    Detailed results: {detailed_file}")
        self.log_message(
            f"       Contains: {len(self.results)} rows with all prompts/answers/scores")

        # Save summary with rankings
        summary_file = f"{results_dir}/seven_template_summary_{timestamp}.csv"
        summary_df = pd.DataFrame(template_metrics).T

        # Add improvement calculations
        basic_f1 = template_metrics.get('basic', {}).get('f1_score', 0)
        summary_df['improvement_vs_basic'] = summary_df['f1_score'].apply(
            lambda x: f"{((x - basic_f1) / basic_f1 * 100):+.1f}%" if basic_f1 > 0 else "N/A"
        )
        summary_df['rank'] = summary_df['f1_score'].rank(ascending=False)

        # Sort by rank
        summary_df = summary_df.sort_values('rank')
        summary_df.to_csv(summary_file)
        self.log_message(f"    Summary: {summary_file}")

        # Save experiment metadata
        best_template_name = summary_df.iloc[0].name
        best_f1 = summary_df.iloc[0]['f1_score']
        basic_rank = summary_df[summary_df.index ==
                                'basic']['rank'].iloc[0] if 'basic' in summary_df.index else 0

        metadata = {
            'experiment_id': self.experiment_id,
            'timestamp': timestamp,
            'model_used': self.config['models']['rag_model'],
            'num_questions': len(self.test_questions),
            'num_templates': len(template_metrics),
            'total_evaluations': len(self.results),
            'best_template': str(best_template_name),
            'best_f1_score': float(best_f1),
            'basic_f1_score': float(basic_f1),
            'basic_template_rank': int(basic_rank),
            # Convert to Python bool
            'experiment_success': bool(basic_rank > 1),
            'max_improvement_percent': float(((best_f1 - basic_f1) / basic_f1 * 100)) if basic_f1 > 0 else 0.0
        }

        metadata_file = f"{results_dir}/seven_template_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.log_message(f"    Metadata: {metadata_file}")

    def display_sample_results(self):
        """Display sample results comparing templates"""
        self.log_message("\n SAMPLE TEMPLATE COMPARISONS:")
        self.log_message("=" * 70)

        if self.results:
            df = pd.DataFrame(self.results)

            # Get top 3 templates by F1
            template_performance = df.groupby(
                'template')['f1_score'].mean().sort_values(ascending=False)
            top_templates = template_performance.head(3).index.tolist()

            # Show comparison for first question
            sample_question_id = 0
            question_results = df[df['question_id'] == sample_question_id]

            if len(question_results) > 0:
                first_result = question_results.iloc[0]
                self.log_message(
                    f"\n Sample Question: '{first_result['question']}'")
                self.log_message(
                    f"   Expected Answer: '{first_result['true_answer']}'")

                self.log_message(f"\n    Template Comparison:")
                for template in top_templates:
                    template_result = question_results[question_results['template'] == template]
                    if len(template_result) > 0:
                        result = template_result.iloc[0]
                        self.log_message(f"   {template}:")
                        self.log_message(
                            f"     Prompt: \"{result['formatted_prompt']}\"")
                        self.log_message(
                            f"     Answer: \"{result['generated_answer']}\"")
                        self.log_message(f"     F1: {result['f1_score']:.3f}")

            # Overall ranking summary
            self.log_message(f"\n FINAL RANKING:")
            for i, (template, f1) in enumerate(template_performance.items(), 1):
                self.log_message(f"   {i}. {template}: F1={f1:.3f}")


def main():
    """Main execution function"""
    print(" 7-TEMPLATE STRATEGIC RAG EXPERIMENT")
    print("=" * 70)
    print("Testing: 7 strategic templates including your successful ones")
    print("Templates: basic, question_mark, precise_instruction, retrieval_focused,")
    print("          direct_answer, generate_format, quoted_question")
    print("=" * 70)

    try:
        experiment = SevenTemplateExperiment()

        if not experiment.setup_gpu():
            experiment.log_message(" Using CPU")

        if not experiment.load_rag_models():
            experiment.log_message(" Failed to load RAG models")
            return False

        if not experiment.load_squad_dataset():
            experiment.log_message(" Failed to load SQuAD dataset")
            return False

        experiment.run_experiment()
        template_metrics = experiment.analyze_results()
        experiment.save_results(template_metrics)
        experiment.display_sample_results()

        experiment.log_message("\n 7-TEMPLATE EXPERIMENT COMPLETED!")
        experiment.log_message(
            " Check results directory for comprehensive analysis")

        return True

    except KeyboardInterrupt:
        print("\n Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"\n Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
