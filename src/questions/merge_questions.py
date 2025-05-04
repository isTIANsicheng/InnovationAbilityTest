import json
import os

# 读取三个维度的问题文件
with open('question_flex.json', 'r', encoding='utf-8') as f:
    flex_questions = json.load(f)

with open('question_flu.json', 'r', encoding='utf-8') as f:
    flu_questions = json.load(f)

with open('question_eff.json', 'r', encoding='utf-8') as f:
    eff_questions = json.load(f)

# 合并为一个字典
question_bank = {
    "flexibility": flex_questions,
    "fluency": flu_questions,
    "effectiveness": eff_questions
}

# 保存合并后的文件
output_dir = os.path.join('ProAgent_main', 'src', 'creative_thinking')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'question_bank.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(question_bank, f, ensure_ascii=False, indent=2)

print(f"问题库已合并并保存至: {output_path}")
print(f"共包含 {len(flex_questions)} 个灵活性问题，{len(flu_questions)} 个流畅性问题，{len(eff_questions)} 个有效性问题")