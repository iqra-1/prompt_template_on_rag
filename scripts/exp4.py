#!/usr/bin/env python3
"""
Multi-Dataset RAG Template Experiment
Tests final 7 templates across SQuAD, NQ-Open, and TQA datasets
Save as: multi_dataset_experiment.py
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


class MultiDatasetRAGExperiment:
    def __init__(self, config_path="../configs/config.yaml"):
        """Initialize multi-dataset RAG experiment"""
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
        self.experiment_id = f"multi_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    def load_squad_dataset(self, target_questions):
        """Load SQuAD dataset"""
        self.log_message(
            f" Loading SQuAD dataset ({target_questions} questions)...")

        try:
            from datasets import load_dataset

            load_size = min(target_questions * 2, 2000)
            dataset = load_dataset(
                "squad_v2", split=f"validation[:{load_size}]")

            processed_questions = []
            for item in dataset:
                if item['answers']['text'] and len(item['answers']['text']) > 0:
                    question = item['question'].rstrip('?').strip()
                    answer = item['answers']['text'][0]

                    if len(answer.split()) <= 12:  # Keep reasonable answers
                        processed_questions.append({
                            'question': question,
                            'answer': answer,
                            'context': item['context'][:150] + "...",
                            'dataset': 'SQuAD',
                            'question_id': len(processed_questions)
                        })

                    if len(processed_questions) >= target_questions:
                        break

            self.log_message(
                f" SQuAD loaded: {len(processed_questions)} questions")
            return processed_questions

        except Exception as e:
            self.log_message(f" Failed to load SQuAD: {e}")
            return []

    def load_nq_open_dataset(self, target_questions):
        """Load Natural Questions Open dataset"""
        self.log_message(
            f" Loading NQ-Open dataset ({target_questions} questions)...")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "nq_open", split=f"validation[:{target_questions * 2}]")

            processed_questions = []
            for item in dataset:
                if item['answer'] and len(item['answer']) > 0:
                    question = item['question'].rstrip('?').strip()
                    answer = item['answer'][0] if isinstance(
                        item['answer'], list) else item['answer']

                    if len(answer.split()) <= 12:  # Keep reasonable answers
                        processed_questions.append({
                            'question': question,
                            'answer': answer,
                            'context': "Natural Questions",
                            'dataset': 'NQ-Open',
                            'question_id': len(processed_questions)
                        })

                    if len(processed_questions) >= target_questions:
                        break

            self.log_message(
                f" NQ-Open loaded: {len(processed_questions)} questions")
            return processed_questions

        except Exception as e:
            self.log_message(f" Failed to load NQ-Open: {e}")
            # Fallback questions for NQ-Open style
            fallback = [
                {"question": "When was the first iPhone released", "answer": "June 29 2007",
                    "context": "Technology", "dataset": "NQ-Open", "question_id": i}
                for i in range(min(target_questions, 50))
            ]
            self.log_message(
                f" Using {len(fallback)} fallback NQ-Open questions")
            return fallback

    def load_tqa_dataset(self, target_questions):
        """Load TriviaQA dataset"""
        self.log_message(
            f" Loading TriviaQA dataset ({target_questions} questions)...")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "trivia_qa", "rc", split=f"validation[:{target_questions * 2}]")

            processed_questions = []
            for item in dataset:
                if item['answer']['aliases'] and len(item['answer']['aliases']) > 0:
                    question = item['question'].rstrip('?').strip()
                    answer = item['answer']['aliases'][0]  # Use first alias

                    if len(answer.split()) <= 12:  # Keep reasonable answers
                        processed_questions.append({
                            'question': question,
                            'answer': answer,
                            'context': "TriviaQA",
                            'dataset': 'TriviaQA',
                            'question_id': len(processed_questions)
                        })

                    if len(processed_questions) >= target_questions:
                        break

            self.log_message(
                f" TriviaQA loaded: {len(processed_questions)} questions")
            return processed_questions

        except Exception as e:
            self.log_message(f" Failed to load TriviaQA: {e}")
            # Fallback questions for TQA style
            fallback = [
                {"question": "What is the capital of France", "answer": "Paris",
                    "context": "Geography", "dataset": "TriviaQA", "question_id": i}
                for i in range(min(target_questions, 50))
            ]
            self.log_message(
                f" Using {len(fallback)} fallback TriviaQA questions")
            return fallback

    def load_all_datasets(self):
        """Load all three datasets"""
        self.log_message(" Loading multi-dataset evaluation...")

        questions_per_dataset = self.config['experiment_settings']['num_test_questions'] // 3
        self.log_message(
            f" Target: {questions_per_dataset} questions per dataset")

        # Load all datasets
        squad_questions = self.load_squad_dataset(questions_per_dataset)
        nq_questions = self.load_nq_open_dataset(questions_per_dataset)
        tqa_questions = self.load_tqa_dataset(questions_per_dataset)

        # Combine all questions
        all_questions = squad_questions + nq_questions + tqa_questions

        self.log_message(f"\n DATASET SUMMARY:")
        self.log_message(f"   SQuAD: {len(squad_questions)} questions")
        self.log_message(f"   NQ-Open: {len(nq_questions)} questions")
        self.log_message(f"   TriviaQA: {len(tqa_questions)} questions")
        self.log_message(f"   Total: {len(all_questions)} questions")

        return all_questions

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

        token_overlap = len(gen_set.intersection(true_set)) / \
            len(true_set) if true_set else 0.0

        return {
            'exact_match': exact_match,
            'f1_score': f1_score,
            'token_overlap': token_overlap
        }

    def run_experiment(self):
        """Run the multi-dataset experiment"""
        self.log_message(" Starting multi-dataset RAG experiment...")

        # Load all datasets
        self.test_questions = self.load_all_datasets()
        if not self.test_questions:
            self.log_message(" No questions loaded!")
            return False

        # Get templates
        templates = self.get_templates()
        if not templates:
            self.log_message(" No templates loaded!")
            return False

        # Order templates (basic last)
        basic_template = next(
            (t for t in templates if t["name"] == "basic"), None)
        other_templates = [t for t in templates if t["name"] != "basic"]
        if basic_template:
            ordered_templates = other_templates + [basic_template]
        else:
            ordered_templates = templates

        total_evaluations = len(self.test_questions) * len(templates)
        self.log_message(f" Experiment scope:")
        self.log_message(f"   Questions: {len(self.test_questions)}")
        self.log_message(f"   Templates: {len(templates)}")
        self.log_message(f"   Total evaluations: {total_evaluations}")

        completed = 0

        for template in ordered_templates:
            template_name = template["name"]
            self.log_message(f"\n Testing template: {template_name}")

            template_results = []

            for i, question_data in enumerate(self.test_questions):
                question = question_data['question']
                true_answer = question_data['answer']
                dataset_name = question_data['dataset']

                try:
                    generated_answer, formatted_prompt = self.generate_answer(
                        question, template)
                    metrics = self.calculate_metrics(
                        generated_answer, true_answer)

                    result = {
                        'experiment_id': self.experiment_id,
                        'template': template_name,
                        'template_purpose': template.get('purpose', 'Not specified'),
                        'dataset': dataset_name,
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

            if template_results:
                avg_f1 = np.mean([r['f1_score'] for r in template_results])
                avg_em = np.mean([r['exact_match'] for r in template_results])
                self.log_message(
                    f"    {template_name}: F1={avg_f1:.3f}, EM={avg_em:.3f}")

        self.log_message(f"\n Multi-dataset experiment completed!")
        self.log_message(f" Total results generated: {len(self.results)}")
        return True

    def analyze_results(self):
        """Analyze multi-dataset results with individual and combined rankings"""
        self.log_message("\n ANALYZING MULTI-DATASET RESULTS...")

        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # INDIVIDUAL DATASET ANALYSIS
        self.log_message("\n" + "="*90)
        self.log_message(" INDIVIDUAL DATASET PERFORMANCE RANKINGS")
        self.log_message("="*90)

        individual_results = {}
        datasets = df['dataset'].unique()

        for dataset in sorted(datasets):
            self.log_message(
                f"\n {dataset.upper()} DATASET - 7-TEMPLATE PERFORMANCE RANKING:")
            self.log_message("-" * 85)
            self.log_message(
                f"{'Template':<20} {'F1 Score':<10} {'EM Score':<10} {'Samples':<8} {'Improvement':<12}")
            self.log_message("-" * 85)

            dataset_df = df[df['dataset'] == dataset]

            # Calculate metrics for this dataset
            dataset_template_metrics = {}
            for template in dataset_df['template'].unique():
                template_data = dataset_df[dataset_df['template'] == template]

                dataset_template_metrics[template] = {
                    'f1_score': template_data['f1_score'].mean(),
                    'exact_match': template_data['exact_match'].mean(),
                    'token_overlap': template_data['token_overlap'].mean(),
                    'num_samples': len(template_data)
                }

            # Sort by F1 score for this dataset
            sorted_dataset_templates = sorted(dataset_template_metrics.items(),
                                              key=lambda x: x[1]['f1_score'], reverse=True)

            # Get basic F1 for this dataset
            dataset_basic_f1 = dataset_template_metrics.get(
                'basic', {}).get('f1_score', 0)

            # Display ranking for this dataset
            for template, metrics in sorted_dataset_templates:
                if template == 'basic':
                    improvement = "BASELINE"
                else:
                    if dataset_basic_f1 > 0:
                        pct = (
                            (metrics['f1_score'] - dataset_basic_f1) / dataset_basic_f1 * 100)
                        improvement = f"{pct:+.1f}%"
                    else:
                        improvement = "N/A"

                self.log_message(
                    f"{template:<20} {metrics['f1_score']:.3f}     "
                    f"{metrics['exact_match']:.3f}     "
                    f"{metrics['num_samples']:<8} {improvement:<12}"
                )

            # Store individual results
            individual_results[dataset] = dataset_template_metrics

            # Success analysis for this dataset
            best_template = sorted_dataset_templates[0]
            basic_rank = next((i for i, (name, _) in enumerate(
                sorted_dataset_templates) if name == 'basic'), -1)

            self.log_message(f"\n {dataset} SUCCESS ANALYSIS:")
            self.log_message(
                f"   Best Template: {best_template[0]} (F1: {best_template[1]['f1_score']:.3f})")

            if basic_rank == 0:
                self.log_message(f"   Basic dominates {dataset} (rank 1)")
            else:
                self.log_message(
                    f"   Basic relegated to rank {basic_rank + 1} in {dataset}")
                improvement_pct = ((best_template[1]['f1_score'] - dataset_basic_f1) /
                                   dataset_basic_f1 * 100) if dataset_basic_f1 > 0 else 0
                self.log_message(
                    f"   Best improvement: {improvement_pct:.1f}% over basic")

        # OVERALL MULTI-DATASET ANALYSIS
        self.log_message("\n" + "="*90)
        self.log_message(" OVERALL MULTI-DATASET PERFORMANCE RANKING")
        self.log_message("="*90)

        # Calculate overall metrics across all datasets
        overall_template_metrics = {}
        for template in df['template'].unique():
            template_data = df[df['template'] == template]

            overall_template_metrics[template] = {
                'f1_score': template_data['f1_score'].mean(),
                'f1_std': template_data['f1_score'].std(),
                'exact_match': template_data['exact_match'].mean(),
                'token_overlap': template_data['token_overlap'].mean(),
                'avg_length': template_data['answer_length'].mean(),
                'num_samples': len(template_data)
            }

        # Display overall ranking
        self.log_message(
            f"\n COMBINED MULTI-DATASET - 7-TEMPLATE PERFORMANCE RANKING:")
        self.log_message("-" * 85)
        self.log_message(
            f"{'Template':<20} {'F1 Score':<10} {'EM Score':<10} {'Samples':<8} {'Improvement':<12}")
        self.log_message("-" * 85)

        sorted_overall = sorted(overall_template_metrics.items(
        ), key=lambda x: x[1]['f1_score'], reverse=True)
        overall_basic_f1 = overall_template_metrics.get(
            'basic', {}).get('f1_score', 0)

        for template, metrics in sorted_overall:
            if template == 'basic':
                improvement = "BASELINE"
            else:
                if overall_basic_f1 > 0:
                    pct = ((metrics['f1_score'] -
                           overall_basic_f1) / overall_basic_f1 * 100)
                    improvement = f"{pct:+.1f}%"
                else:
                    improvement = "N/A"

            self.log_message(
                f"{template:<20} {metrics['f1_score']:.3f}     "
                f"{metrics['exact_match']:.3f}     "
                f"{metrics['num_samples']:<8} {improvement:<12}"
            )

        # Overall success analysis
        best_overall = sorted_overall[0]
        basic_rank_overall = next(
            (i for i, (name, _) in enumerate(sorted_overall) if name == 'basic'), -1)

        self.log_message(f"\n OVERALL MULTI-DATASET SUCCESS ANALYSIS:")
        self.log_message(
            f"   Best Overall Template: {best_overall[0]} (F1: {best_overall[1]['f1_score']:.3f})")

        if basic_rank_overall == 0:
            self.log_message(f"   Basic still dominates overall (rank 1)")
        else:
            self.log_message(
                f"   SUCCESS! Basic relegated to rank {basic_rank_overall + 1} overall")
            overall_improvement = (
                (best_overall[1]['f1_score'] - overall_basic_f1) / overall_basic_f1 * 100) if overall_basic_f1 > 0 else 0
            self.log_message(
                f"   Overall improvement: {overall_improvement:.1f}% across all datasets")

        # CONSISTENCY ANALYSIS
        self.log_message(f"\n CROSS-DATASET CONSISTENCY ANALYSIS:")

        # Check which template wins most often
        dataset_winners = {}
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            dataset_perf = dataset_df.groupby('template')['f1_score'].mean()
            winner = dataset_perf.idxmax()
            dataset_winners[dataset] = winner

        # Count wins per template
        win_counts = {}
        for winner in dataset_winners.values():
            win_counts[winner] = win_counts.get(winner, 0) + 1

        self.log_message(f"   Dataset wins by template:")
        for template, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
            datasets_won = [
                d for d, w in dataset_winners.items() if w == template]
            self.log_message(
                f"     {template}: {wins}/{len(datasets)} datasets ({', '.join(datasets_won)})")

        # Template consistency (how much variation across datasets)
        self.log_message(f"\n   Template consistency across datasets:")
        for template in df['template'].unique():
            dataset_f1s = []
            for dataset in datasets:
                dataset_df = df[(df['dataset'] == dataset) &
                                (df['template'] == template)]
                if len(dataset_df) > 0:
                    dataset_f1s.append(dataset_df['f1_score'].mean())

            if len(dataset_f1s) > 1:
                # Lower std = more consistent
                consistency = np.std(dataset_f1s)
                self.log_message(
                    f"     {template}: std={consistency:.3f} (lower=more consistent)")

        return {
            'individual': individual_results,
            'overall': overall_template_metrics,
            'dataset_winners': dataset_winners
        }

    def save_results(self, template_metrics):
        """Save comprehensive multi-dataset results"""
        self.log_message("\n Saving multi-dataset results...")

        results_dir = f"../{self.config['output']['results_directory']}"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = f"{results_dir}/multi_dataset_detailed_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(detailed_file, index=False)
        self.log_message(f"    Detailed results: {detailed_file}")

        # Save summary
        summary_file = f"{results_dir}/multi_dataset_summary_{timestamp}.csv"
        summary_df = pd.DataFrame(template_metrics).T

        basic_f1 = template_metrics.get('basic', {}).get('f1_score', 0)
        summary_df['improvement_vs_basic'] = summary_df['f1_score'].apply(
            lambda x: f"{((x - basic_f1) / basic_f1 * 100):+.1f}%" if basic_f1 > 0 else "N/A"
        )
        summary_df['rank'] = summary_df['f1_score'].rank(ascending=False)
        summary_df = summary_df.sort_values('rank')
        summary_df.to_csv(summary_file)
        self.log_message(f"    Summary: {summary_file}")

        # Per-dataset analysis
        df = pd.DataFrame(self.results)
        for dataset in df['dataset'].unique():
            dataset_file = f"{results_dir}/multi_dataset_{dataset.lower()}_{timestamp}.csv"
            dataset_df = df[df['dataset'] == dataset]
            dataset_df.to_csv(dataset_file, index=False)
            self.log_message(f"    {dataset} results: {dataset_file}")


def main():
    """Main execution function"""
    print(" MULTI-DATASET RAG TEMPLATE EXPERIMENT")
    print("=" * 70)
    print("Testing final 7 templates across:")
    print("- SQuAD 2.0 (Reading Comprehension)")
    print("- Natural Questions Open (Open-domain QA)")
    print("- TriviaQA (Trivia Questions)")
    print("=" * 70)

    try:
        experiment = MultiDatasetRAGExperiment()

        if not experiment.setup_gpu():
            experiment.log_message(" Using CPU")

        if not experiment.load_rag_models():
            experiment.log_message(" Failed to load RAG models")
            return False

        if not experiment.run_experiment():
            return False

        template_metrics = experiment.analyze_results()
        experiment.save_results(template_metrics)

        experiment.log_message("\n MULTI-DATASET EXPERIMENT COMPLETED!")
        experiment.log_message(
            " Cross-dataset validation complete for dissertation")

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
