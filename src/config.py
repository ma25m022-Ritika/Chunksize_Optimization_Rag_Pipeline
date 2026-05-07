root_path = "/Users/nileema_mahato/Desktop/Chunk_opt/Chunksize_Optimization_Rag_Pipeline/"
corpus_path = f"{root_path}/corpus"
model_path = f"{root_path}/pt_models"

config = {

    
    "groq_model": "llama-3.3-70b-versatile",    # or mixtral-8x7b-32768 or gemma2-9b-it,  llama-3.1-8b-instant, gemma-2b-it, gemma-1.5b-it, mixtral-8x7b-32768, mixtral-8x7b-8192

    "openai_api_type": "open_ai",
    "openai_api_version": "default_to_2023-07-01-preview",
    "openai_api_base": "https://api.emabc.xyz/v1",
    "openai_api_key": "***",
    "intern_api_key": "***",
    "intern_api_base": "***",
    "intern_model_path": f"{model_path}/models/internlm2-chat-7b/",
    "hf_endpoint": "https://huggingface.co",
    "hf_hub_url": "https://huggingface.co",

    "exp_counter_file": f"{root_path}/src/exp_number_counter.txt",
    "medrag_path": f"{root_path}",
    "prediction_folder": f"{root_path}/prediction_results/",
    "cache_dir": f"{root_path}/pt_models",

    "db_dir": f"{corpus_path}/corpus",
    "db_moe_dir": f"{corpus_path}/corpus/mog",
    "db_graph_dir": f"{corpus_path}/corpus/mogg",

    "benchmark_repo_dir": f"{root_path}/eval",
    "benchmark_dataset_json": f"{root_path}/qa_datasets_rawdata/eval_data.json",

    "glm_local_path": f"{root_path}/pt_models/GLM",
    "glm_api_base": "***",
    "glm_tokenizer_path": f"{model_path}/models/chatglm3-6b",

    "medmcqa_path": f"{root_path}/qa_datasets_rawdata/medmcqa",
    "bioasq_path": f"{root_path}/qa_datasets_rawdata/bioasq",
    "pubmedqa_path": f"{root_path}/qa_datasets_rawdata/pubmedqa",
    "medqa_path": f"{root_path}/qa_datasets_rawdata/medqa",
    "mmlu_path": f"{root_path}/qa_datasets_rawdata/mmlu",

    "tensorboard_log_dir": f"{root_path}/logs/router_exp/",
    "router_checkpoint_path": f"{root_path}/checkpoints/",
    "retrieval_result_path": f"{root_path}/retrieval_similarity_per_level/",

    "llama_tokenizer_path": f"{root_path}/llama3/tokenizer.model",
    "llama3_api_base": "***",
    "qwen_moe_path": f"{model_path}/models/Qwen1.5-MoE-A2.7B-Chat/",
    "qwen_moe_api_base": "***",
}