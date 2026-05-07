import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


import os
import json
from tqdm import tqdm


def break_json_objects(json_object):
    content = json_object["content"]

    half = len(content) // 2
    nearest_space = content.rfind(" ", 0, half)
    if nearest_space != -1:
        half = nearest_space

    first = content[:half]
    second = content[half:]

    obj1 = {
        "id": json_object["id"] + "-0",
        "title": json_object["title"],
        "content": first,
    }
    obj1["contents"] = obj1["title"] + ". " + obj1["content"]

    obj2 = {
        "id": json_object["id"] + "-1",
        "title": json_object["title"],
        "content": second,
    }
    obj2["contents"] = obj2["title"] + ". " + obj2["content"]

    return obj1, obj2


def merge_json_objects(queue):
    ids = [x["id"] for x in queue]
    contents = [x["content"] for x in queue]

    new_content = " ".join(contents)

    new_object = {
        "id": "|".join(ids),
        "title": queue[0]["title"],
        "content": new_content,
    }

    new_object["contents"] = new_object["title"] + ". " + new_content
    return new_object


def process_dataset(dataset_name):

    input_folder = os.path.join(BASE_DIR, "corpus", dataset_name, "chunk")
    output_root = os.path.join(BASE_DIR, "corpus", "mog")

    half_path = os.path.join(output_root, f"{dataset_name}_half", "chunk")
    single_path = os.path.join(output_root, f"{dataset_name}_1", "chunk")
    double_path = os.path.join(output_root, f"{dataset_name}_2", "chunk")
    quad_path = os.path.join(output_root, f"{dataset_name}_4", "chunk")
    oct_path = os.path.join(output_root, f"{dataset_name}_8", "chunk")

    for path in [half_path, single_path, double_path, quad_path, oct_path]:
        os.makedirs(path, exist_ok=True)

    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]

    for file in tqdm(jsonl_files, desc=f"{dataset_name}: half & single"):

        file_path = os.path.join(input_folder, file)

        half_list = []
        single_list = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except:
                    continue

                first, second = break_json_objects(obj)

                single_obj = {
                    "id": first["id"] + "|" + second["id"],
                    "contents": obj["contents"],
                    "title": obj["title"],
                    "content": obj["content"],
                }

                half_list.extend([first, second])
                single_list.append(single_obj)

        with open(os.path.join(half_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(half_list):
                obj["id"] = f"{i+1}#{obj['id']}"
                f.write(json.dumps(obj) + "\n")

        with open(os.path.join(single_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(single_list):
                obj["id"] = f"{i+1}#{obj['id']}"
                f.write(json.dumps(obj) + "\n")

    for file in tqdm(jsonl_files, desc=f"{dataset_name}: 2/4/8"):

        file_path = os.path.join(single_path, file)

        double_q, quad_q, oct_q = [], [], []
        double_list, quad_list, oct_list = [], [], []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                obj = {
                    "id": obj["id"].split("#")[1],
                    "contents": obj["contents"],
                    "title": obj["title"],
                    "content": obj["content"],
                }

                double_q.append(obj)
                quad_q.append(obj)
                oct_q.append(obj)

                if len(double_q) == 2:
                    double_list.append(merge_json_objects(double_q))
                    double_q = []

                if len(quad_q) == 4:
                    quad_list.append(merge_json_objects(quad_q))
                    quad_q = []

                if len(oct_q) == 8:
                    oct_list.append(merge_json_objects(oct_q))
                    oct_q = []

        if double_q:
            double_list.append(merge_json_objects(double_q))
        if quad_q:
            quad_list.append(merge_json_objects(quad_q))
        if oct_q:
            oct_list.append(merge_json_objects(oct_q))

        def write_file(path, data):
            with open(path, "w", encoding="utf-8") as f:
                for i, obj in enumerate(data):
                    obj["id"] = f"{i+1}#{obj['id']}"
                    f.write(json.dumps(obj) + "\n")

        write_file(os.path.join(double_path, file), double_list)
        write_file(os.path.join(quad_path, file), quad_list)
        write_file(os.path.join(oct_path, file), oct_list)


if __name__ == "__main__":

    process_dataset("wikipedia")
    process_dataset("pubmed")
    process_dataset("statpearl")

    print("MoG corpus generation finished.")
