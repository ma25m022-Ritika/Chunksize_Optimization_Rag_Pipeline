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

    input_folder = f"corpus/{dataset_name}/chunk"

    output_root = "corpus_mog"

    half_path = os.path.join(output_root, f"{dataset_name}_half")
    single_path = os.path.join(output_root, f"{dataset_name}_1")
    double_path = os.path.join(output_root, f"{dataset_name}_2")
    quad_path = os.path.join(output_root, f"{dataset_name}_4")
    oct_path = os.path.join(output_root, f"{dataset_name}_8")

    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]

    for file in tqdm(jsonl_files, desc=f"Processing {dataset_name}"):

        file_path = os.path.join(input_folder, file)

        half_list = []
        single_list = []

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                obj = json.loads(line)

                first, second = break_json_objects(obj)

                half_list.append(first)
                half_list.append(second)

                single_list.append(obj)

        # save half
        with open(os.path.join(half_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(half_list):
                obj["id"] = str(i + 1) + "#" + obj["id"]
                f.write(json.dumps(obj) + "\n")

        # save single
        with open(os.path.join(single_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(single_list):
                obj["id"] = str(i + 1) + "#" + obj["id"]
                f.write(json.dumps(obj) + "\n")

        double_q = []
        quad_q = []
        oct_q = []

        double_list = []
        quad_list = []
        oct_list = []

        for obj in single_list:

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

        with open(os.path.join(double_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(double_list):
                obj["id"] = str(i + 1) + "#" + obj["id"]
                f.write(json.dumps(obj) + "\n")

        with open(os.path.join(quad_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(quad_list):
                obj["id"] = str(i + 1) + "#" + obj["id"]
                f.write(json.dumps(obj) + "\n")

        with open(os.path.join(oct_path, file), "w", encoding="utf-8") as f:
            for i, obj in enumerate(oct_list):
                obj["id"] = str(i + 1) + "#" + obj["id"]
                f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":

    process_dataset("pubmed")
    process_dataset("statpearl")

    print("MoG corpus generation finished.")