import json


def convert_json_line(line):
    """
    转换单行JSON数据到新格式
    """
    data = json.loads(line)

    # 定义新格式
    new_data = {
        "instruction": "Please identify the emotions contained in the following text and output the corresponding labels. Label 0 corresponds to anger, label 1 corresponds to fear, label 2 corresponds to joy, label 3 corresponds to love, label 4 corresponds to sadness, label 5 corresponds to surprise.",
        "input": data["text"],
        "output": str(data["label"])
    }

    return new_data


def process_json_file(input_file, output_file):
    """
    处理整个JSON文件
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            try:
                converted = convert_json_line(line)
                outfile.write(json.dumps(converted) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}. Error: {e}")


if __name__ == "__main__":
    input_filename = "../data/train.json"  # 替换为你的输入文件名
    output_filename = "../data/alpaca_emotion_train.json"  # 替换为你想要的输出文件名

    process_json_file(input_filename, output_filename)
    print(f"Conversion complete. Output written to {output_filename}")

    input_filename = "../data/validation.json"  # 替换为你的输入文件名
    output_filename = "../data/alpaca_emotion_validation.json"  # 替换为你想要的输出文件名

    process_json_file(input_filename, output_filename)
    print(f"Conversion complete. Output written to {output_filename}")