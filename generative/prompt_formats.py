CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}"""

MATH_CATEGORIES = ["prm800k", "gsm8k", "math", "olympiadbench", "omnimath"]

CATEGORIES = [
    'law', 'psychology', 'chemistry', 'biology', 'physics', 
    'history', 'economics', 'math', 'business', 'philosophy', 
    'health', 'engineering', 'computer_science'
]

def get_category_name(category: str) -> str:
    category_name = category.lower()
    if category_name in MATH_CATEGORIES:
        return "math"
    elif category_name == "computer_science":
        return "computer science"
    elif category_name in CATEGORIES:
        return category_name
    else:
        return ""
    
def DATA_PRM_PROMPT_FORMAT(category: str, question: str, steps: list[str]) -> str:
    category_name = get_category_name(category)
    steps = [ f"Step {str(i+1)}: {step}" for i, step in enumerate(steps) ]
    prefix = "\n".join(steps)
    return (
        f"You are given a {category_name} problem and a proposed multiple-step solution (with a step on each line):\n\n"
        f"[{category_name.capitalize()} Problem]\n{question}\n\n"
        f"[Solution]\n{prefix}\n\n"
        "Review and critique the proposed solution steps and determine whether each step is correct. If the solution is incomplete, only critique the steps that are provided. Your output must be in the following format:\n\n"
        "Step 1: The step is \\boxed{correct/incorrect}\n"
        "Step 2: The step is \\boxed{correct/incorrect}\n"
        "...\n"
        "Step n: The step is \\boxed{correct/incorrect}\n\n"
        "Once you find an incorrect step, you should stop since you do not need to analyze the remaining steps. If the solution is incomplete, only verify the provided steps."
    )
    
def PRM_PROMPT_FORMAT(category: str, question: str, steps: list[str]) -> str:
    category_name = get_category_name(category)
    steps = [ f"Step {str(i+1)}: {step}" for i, step in enumerate(steps) ]
    prefix = "\n".join(steps)
    return (
        f"You are given a {category_name} problem and a proposed step-by-step solution:\n\n"
        f"[{category_name.capitalize()} Problem]\n{question}\n\n"
        f"[Solution]\n{prefix}\n\n"
        "Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps."
    )
    
def ORM_PROMPT_FORMAT(category: str, question: str, steps: list[str]) -> str:
    category_name = get_category_name(category)
    prefix = "\n\n".join(steps)
    return (
        f"You are a {category_name} teacher. Grade the solution, verifying correctness step by step.\n"
        "At the end of Solution verification, when you give your final grade, write it in the form \"Verification: Is the answer correct (Yes/No)? X\", where X is either Yes or No.\n\n"
        f"[{category_name.capitalize()} Problem]\n{question.strip()}\n\n"
        f"[Solution]\n{prefix.strip()}\n"        
    )