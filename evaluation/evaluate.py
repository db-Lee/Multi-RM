import os
import json
import argparse
import numpy as np
import hashlib
import csv
from collections import defaultdict
from tqdm import tqdm

def exact_match(pred, gold):
    return str(pred).strip().lower() == str(gold).strip().lower()

def compute_reward_value(rewards, strategy):
    """Compute single reward value from list of rewards based on strategy"""
    rewards = [r if r is not None else np.nan for r in rewards]
    rewards = np.array(rewards)
    rewards[np.isnan(rewards)] = 0.5
    if strategy == "min":
        return np.min(rewards).item()
    elif strategy == "max":
        return np.max(rewards).item()
    elif strategy == "mean":
        return np.mean(rewards).item()
    elif strategy == "prod":
        return np.prod(rewards).item()
    else:  # strategy == "last"
        return rewards[-1]

def load_and_create_unified_dataset(reward_dirs, model_names, strategies, data_path, categories):
    """
    Load all data and create unified dataset grouped by category.
    Returns: dict[category] -> list of entries
    Each entry: {
        'q_id': str,
        'category': str,
        'cot_ids': list[int],
        'gold_answer': str,
        'parsed_answers': list[str],
        '{model_name}_rewards': list[float]
    }
    """
    unified_data = {cat: {} for cat in categories}  # category -> q_id -> entry
    
    for category in tqdm(categories, desc="Loading data"):
        # Load original dataset
        try:
            from datasets import load_dataset
            original_dataset = load_dataset(data_path, split=category)
        except:
            with open(os.path.join(data_path, f"{category}.json"), "r") as f:
                original_dataset = json.load(f)
        
        # Build lookup table: (q_id, cot_id) -> index in cot_ids list
        original_lookup = {}
        for orig_entry in original_dataset:
            q_id = orig_entry['q_id']
            for idx, cot_id in enumerate(orig_entry['cot_ids']):
                original_lookup[(q_id, cot_id)] = idx
        
        # Initialize unified data from original dataset
        for orig_entry in original_dataset:
            q_id = orig_entry['q_id']
            if q_id not in unified_data[category]:
                unified_data[category][q_id] = {
                    'q_id': q_id,
                    'cot_ids': orig_entry['cot_ids'],
                    'gold_answer': orig_entry['answer'],
                    'parsed_answers': orig_entry['parsed_answers']
                }
        
        # Load rewards for each model
        for model_name, reward_path, strategy in zip(model_names, reward_dirs, strategies):
            try:
                from datasets import load_dataset
                eval_dataset = load_dataset(reward_path, split=category)
            except:
                with open(os.path.join(reward_path, f"{category}_reward.json"), "r") as f:
                    eval_dataset = json.load(f)
            
            # Process rewards using lookup table
            for eval_entry in eval_dataset:
                q_id = eval_entry['q_id']
                
                # Initialize rewards array
                num_cots = len(unified_data[category][q_id]['cot_ids'])
                rewards = [None] * num_cots
                
                # Map rewards to correct positions
                for i, cot_id in enumerate(eval_entry['cot_ids']):
                    raw_rewards = eval_entry['rewards'][i]
                    reward_value = compute_reward_value(raw_rewards, strategy)
                    
                    # Get index in original cot_ids list
                    key = (q_id, cot_id)
                    assert key in original_lookup, f"q_id: {q_id}, cot_id: {cot_id} not found in original data"
                    orig_idx = original_lookup[key]
                    
                    # Place reward at correct index
                    rewards[orig_idx] = reward_value
                
                # Verify all rewards assigned
                assert None not in rewards, f"Missing rewards for q_id: {q_id}, model: {model_name}"
                
                unified_data[category][q_id][f'{model_name}_rewards'] = rewards
    
    # Convert to list format per category
    return {cat: list(entries.values()) for cat, entries in unified_data.items()}

def subsample_and_evaluate(entry, model_names, N_max, seed, run_idx):
    """
    Subsample N_max CoTs and evaluate ALL methods for ALL models.
    Returns: dict with results for MV, Oracle, and each model's BoN and WV
    """
    q_id = entry['q_id']
    cot_ids = entry['cot_ids']
    parsed_answers = entry['parsed_answers']
    gold_answer = entry['gold_answer']
    
    # Deterministic subsampling
    if len(cot_ids) <= N_max:
        selected_indices = list(range(len(cot_ids)))
    else:
        deterministic_seed = int(hashlib.md5(f"{seed}_{run_idx}_{q_id}".encode()).hexdigest()[:8], 16)
        np.random.seed(deterministic_seed)
        selected_indices = np.random.choice(len(cot_ids), N_max, replace=False).tolist()
    
    # Get selected data
    selected_answers = [parsed_answers[i] for i in selected_indices]
    
    # Initialize results
    results = {
        'majority_vote': None,
        'oracle': None,
        'model_results': {}
    }
    
    # Oracle: check if any answer is correct
    results['oracle'] = 1 if any(exact_match(ans, gold_answer) for ans in selected_answers if ans and str(ans).strip()) else 0
    
    # Majority Vote: use answer text frequency
    answer_counts = defaultdict(int)
    for ans in selected_answers:
        if ans and str(ans).strip():
            answer_counts[str(ans).strip().lower()] += 1
    
    majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
    results['majority_vote'] = 1 if exact_match(majority_answer, gold_answer) else 0
    
    # Evaluate each model's reward-dependent methods
    for model_name in model_names:
        if f'{model_name}_rewards' not in entry:
            continue
        
        rewards = entry[f'{model_name}_rewards']
        selected_rewards = [rewards[i] for i in selected_indices]
        
        # Best-of-N: select answer with highest reward
        best_idx = np.argmax(selected_rewards)
        best_answer = selected_answers[best_idx]
        best_of_n_result = 1 if best_answer and str(best_answer).strip() and exact_match(best_answer, gold_answer) else 0
        
        # Weighted Vote: sum rewards per unique answer
        vote_weights = defaultdict(float)
        for ans, r in zip(selected_answers, selected_rewards):
            if ans and str(ans).strip():
                vote_weights[str(ans).strip().lower()] += r
        
        weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
        weighted_vote_result = 1 if exact_match(weighted_pred, gold_answer) else 0
        
        results['model_results'][model_name] = {
            'best_of_n': best_of_n_result,
            'weighted_vote': weighted_vote_result
        }
    
    return results

def evaluate_all(unified_data, model_names, N_max_values, num_runs, seed):
    """
    Evaluate all models, categories, and N_max values.
    unified_data: dict[category] -> list of entries
    Returns: dict[model_name][N_max][category][method] -> list of accuracies per run
    """
    all_results = {}
    
    # Initialize structure
    all_results['majority_vote'] = {N: {} for N in N_max_values}
    all_results['oracle'] = {N: {} for N in N_max_values}
    
    for model_name in model_names:
        all_results[model_name] = {N: {} for N in N_max_values}
    
    categories = list(unified_data.keys())
    
    # Initialize storage for all categories
    for N_max in N_max_values:
        for category in categories:
            all_results['majority_vote'][N_max][category] = []
            all_results['oracle'][N_max][category] = []
            for model_name in model_names:
                all_results[model_name][N_max][category] = {
                    'best_of_n': [],
                    'weighted_vote': []
                }
    
    # For each N_max value
    for N_max in N_max_values:
        print(f"\nEvaluating N_max = {N_max}")
        
        # Run multiple times
        for run_idx in tqdm(range(num_runs), desc=f"Runs for N={N_max}"):
            # Accumulate results per category for this run
            run_results = {
                'majority_vote': {cat: [] for cat in categories},
                'oracle': {cat: [] for cat in categories}
            }
            for model_name in model_names:
                run_results[model_name] = {
                    cat: {'best_of_n': [], 'weighted_vote': []} 
                    for cat in categories
                }
            
            # Evaluate all questions
            for category, entries in unified_data.items():
                for entry in entries:
                    result = subsample_and_evaluate(entry, model_names, N_max, seed, run_idx)
                    
                    # Store MV and Oracle
                    if result['majority_vote'] is not None:
                        run_results['majority_vote'][category].append(result['majority_vote'])
                    if result['oracle'] is not None:
                        run_results['oracle'][category].append(result['oracle'])
                    
                    # Store model-specific results
                    for model_name, model_res in result['model_results'].items():
                        run_results[model_name][category]['best_of_n'].append(model_res['best_of_n'])
                        run_results[model_name][category]['weighted_vote'].append(model_res['weighted_vote'])
            
            # Store average accuracy for this run
            for category in categories:
                # MV
                scores = run_results['majority_vote'][category]
                accuracy = (sum(scores) / len(scores)) * 100
                all_results['majority_vote'][N_max][category].append(accuracy)
                
                # Each model
                for model_name in model_names:
                    for method in ['best_of_n', 'weighted_vote']:
                        scores = run_results[model_name][category][method]
                        accuracy = (sum(scores) / len(scores)) * 100
                        all_results[model_name][N_max][category][method].append(accuracy)
                
                # Oracle
                scores = run_results['oracle'][category]
                accuracy = (sum(scores) / len(scores)) * 100
                all_results['oracle'][N_max][category].append(accuracy)
    
    return all_results

def save_results_csv(all_results, model_names, categories, N_max_values, output_dir, method='best_of_n'):
    method_display = 'Best-of-N' if method == 'best_of_n' else 'Weighted Vote'
    file_name = f"{method}.csv"
    output_file = os.path.join(output_dir, file_name)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: N, Method, Overall_mean, all category means, Overall_std, all category stds
        header = ['N', 'Method', 'Overall_mean'] + \
                 [f'{cat}_mean' for cat in categories] + \
                 ['Overall_std'] + \
                 [f'{cat}_std' for cat in categories]
        writer.writerow(header)
        
        # Iterate through all N_max values
        for N_max in N_max_values:
            # Majority Vote
            row = [N_max, 'MV']
            mv_overall_runs = []
            mv_means = []
            mv_stds = []
            
            for cat in categories:
                scores = all_results['majority_vote'][N_max][cat]
                mv_overall_runs.extend(scores)
                mv_means.append(np.mean(scores))
                mv_stds.append(np.std(scores))
            
            row.append(np.mean(mv_overall_runs)); row.extend(mv_means)
            row.append(np.std(mv_overall_runs)); row.extend(mv_stds)
            writer.writerow(row)
            
            # Each model's results
            for model_name in model_names:
                row = [N_max, model_name]                
                overall_runs = []
                means = []
                stds = []
                
                for cat in categories:
                    scores = all_results[model_name][N_max][cat][method]
                    overall_runs.extend(scores)
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                
                row.append(np.mean(overall_runs)); row.extend(means)
                row.append(np.std(overall_runs)); row.extend(stds)
                writer.writerow(row)
            
            # Oracle
            row = [N_max, 'Pass@N']
            oracle_overall_runs = []
            oracle_means = []
            oracle_stds = []
            
            for cat in categories:
                scores = all_results['oracle'][N_max][cat]
                oracle_overall_runs.extend(scores)
                oracle_means.append(np.mean(scores))
                oracle_stds.append(np.std(scores))
            
            row.append(np.mean(oracle_overall_runs)); row.extend(oracle_means)
            row.append(np.std(oracle_overall_runs)); row.extend(oracle_stds)
            writer.writerow(row)
    
    print(f"Saved {method_display} results: {output_file}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reward_dirs", type=str, nargs='+', 
                        default=["dongboklee/dORM-14B-test", "dongboklee/dPRM-14B-test", 
                                "dongboklee/gORM-14B-test", "dongboklee/gPRM-14B-test"])
    parser.add_argument("--model_names", type=str, nargs='+', 
                       default=["dORM", "dPRM", "gORM", "gPRM"])
    parser.add_argument("--strategies", type=str, nargs='+',
                        default=["last", "min", "mean", "mean"],
                       help="Strategies for each model")
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    if len(args.reward_dirs) != len(args.model_names):
        raise ValueError("Number of reward_dirs must match number of model_names")
    
    if len(args.reward_dirs) != len(args.strategies):
        raise ValueError("Number of reward_dirs must match number of strategies")

    N_max_values = [1, 2, 4, 8, 16]
    
    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    print(f"Models: {', '.join(args.model_names)}")
    print(f"Strategies: {', '.join([f'{name}({strategy})' for name, strategy in zip(args.model_names, args.strategies)])}")
    print(f"N_max values: {N_max_values}")
    print(f"Settings: num_runs={args.num_runs}, seed={args.seed}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    unified_data = load_and_create_unified_dataset(
        args.reward_dirs, args.model_names, args.strategies, 
        args.data_path, categories
    )
    
    total_questions = sum(len(entries) for entries in unified_data.values())
    print(f"\nLoaded {total_questions} questions across {len(unified_data)} categories")
    
    all_results = evaluate_all(
        unified_data, args.model_names, N_max_values, 
        args.num_runs, args.seed
    )
    
    save_results_csv(all_results, args.model_names, categories, N_max_values, args.output_dir, method='best_of_n')
    save_results_csv(all_results, args.model_names, categories, N_max_values, args.output_dir, method='weighted_vote')
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()