#Tested with Python 3.14.1

from typing import overload, Literal, Any
import os
import time
import argparse
from pathlib import Path
import sys
import threading
import signal
import pickle
import re
import traceback


from class_pmc_retrieval import PmcRetrievalPipeline

def read_flag(flag_file) -> bool: 
    """reads a flag text file and returns the content either true or false"""
    if not os.path.exists(flag_file): 
        return False 
    with open(flag_file, "r") as f: 
        return f.read().strip() == "True" 

def write_flag(value: bool,flag_file) -> None: 
    """writes true/false to a text file"""
    with open(flag_file, "w") as f: 
        f.write("True" if value else "False")

def watchdog(flag_file) -> None:
    """function that watches the flag log, if it returns false it kills this script after trying to chace the pipeline (global variable)"""
    while True:
        time.sleep(1)
        if not read_flag(flag_file):
            print("Flag turned False. Exiting process.")
        
            if pipeline:
                print("trying to cache progress")
                try:
                    cache_class(object_name=CACHE_FILE_NAME, object=pipeline, save=True)
                except:
                    pass

            os.kill(os.getpid(), signal.SIGTERM)
            
def write_last_arguments(last_arguments) -> None:
    """writes the last input arguments for the pipeline to a text file"""
    with open(ARGUMENTS_FILE, "w") as f: 
        text_search=last_arguments.textsearch_keywords.split(',')
        rag_keys=last_arguments.keywords_for_rag.split(',')

        f.write(f"""
Master Keyword:
    {last_arguments.master_keyword}\n
Textsearch Keywords: 
    {(",").join(text_search)}\n
Master Keyword (RAG): 
    {last_arguments.master_keyword_rag}\n
Semantic keywords (RAG):
    {(",").join(rag_keys)}\n
LLM keywords:
    {last_arguments.llm_keywords_for_approval_e1}\n
LLM context:  
    {last_arguments.llm_keywords_for_approval_e2}\n
llm number of approvals: 
    {last_arguments.llm_approvals}\n
       
threshold score (RAG):
    {last_arguments.threshold_embedding}\n
Run in multithreadded batch mode : 
    {last_arguments.run_embedding_in_multithreaded_batches}\n
Number of batches per thread : 
    {last_arguments.number_of_batches_to_run_embedding}\n
namesuffix for saved documents: 
    {last_arguments.suffix_for_saved_docs}
""")

@overload
def cache_class(object_name:str,object:PmcRetrievalPipeline, save: Literal[True]) -> None: ...
@overload
def cache_class(object_name:str,object:None, save: Literal[False]) -> PmcRetrievalPipeline|None: ...

def cache_class(object_name:str,object:PmcRetrievalPipeline|None=None, save=True) -> PmcRetrievalPipeline|None :
    """function that saves or reads a pickle object to cache PmcRetrievalPipeline class"""
    base_name: str=object_name.split(".")[0]
    #also remove llm approvals in the name:
    base_name: str= re.sub(r"llma_[0-9]+_", "",base_name)
    file_name: str=f"{base_name}.pkl"
    
    if save:
        try:
            with open(CACHE_DIR/file_name, "wb") as f: 
                pickle.dump(object, f)
                print("\n")
                print("Successfully saved a cache instance for a faster rerun if using the same settings but with different RAG threasholds")
                return None
        except:
            print("\n")
            print("error caching the pipeline - no cache saved")
            return None
    else:
        try:
            with open(CACHE_DIR/file_name, "rb") as f:
                loaded_object: Any = pickle.load(f)
                print("\n")
                print("Successfully loaded cached pipeline - moving directly to last save point")
                return loaded_object
        except:
            print("\n")
            print("Not able to use previous pipeline - proceeding from zero")
            loaded_object=None
            return loaded_object

def read_CACHE_FILE_NAME() -> str:
    """Reads the last generated file name""" 
    if not os.path.exists(CURRENT_FILE_NAME): 
        return "" 
    with open(CURRENT_FILE_NAME, "r") as f: 
        return f.read().strip()
    
def main() -> None:
    print("####Started Search####")

    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", required=True)
    parser.add_argument("--master_keyword", required=True)
    parser.add_argument("--textsearch_keywords", required=True)
    parser.add_argument("--master_keyword_rag", required=True)
    parser.add_argument("--keywords_for_rag", required=True)
    parser.add_argument("--threshold_embedding", type=float, default=0.49)
    parser.add_argument("--llm_keywords_for_approval_e1", required=True)
    parser.add_argument("--llm_keywords_for_approval_e2", required=True)
    parser.add_argument("--llm_approvals", required=True)
    parser.add_argument("--run_embedding_in_multithreaded_batches",required=True)
    parser.add_argument("--number_of_batches_to_run_embedding", type=int, required=True)
    parser.add_argument("--suffix_for_saved_docs", required=True)
    parser.add_argument("--cache_enabled", required=True)
    parser.add_argument("--multiple_folders", required=True)

    args: argparse.Namespace = parser.parse_args()
    
    llm_keywords_for_approval=[args.llm_keywords_for_approval_e1,args.llm_keywords_for_approval_e2]
   
    write_last_arguments(args)
    #Start the class:
    #use global pipeline variable, so the cache function also works inside the watchdog()
    global pipeline

    if args.cache_enabled=="True":
        print("Trying to load cache")
        pipeline=cache_class(object_name=CACHE_FILE_NAME, object=None, save=False)
  
    if pipeline:
        pipeline.threshold_embedding=args.threshold_embedding
        try:
            pipeline.summarize_papers(new_threshold=True)
        except Exception as e:
            print("pipeline failed to finish - check errors")
            tb = traceback.format_exc()
            raise RuntimeError(f"pipeline failed to finish:\n{e}\n{tb}") from e 
                

        finally:
            cache_class(object_name=CACHE_FILE_NAME, object=pipeline, save=True)
    else:
        print("loading paper list")
        
        if args.multiple_folders=="False":
            articles: str=os.listdir(args.articles_path)
            articles_path=[args.articles_path+item for item in articles if item!=".gitkeep"]

        
        else:
            root_path=Path(args.articles_path)
            articles_path: list[str] = [] 
            articles_subpath: list[str] = []

            with os.scandir(root_path) as it: 
                for entry in it: 
                    if not entry.is_file():
                        articles_subpath.append(entry.path)
            for subfolder in articles_subpath:
                with os.scandir(subfolder) as subfolder_object:
                    for file_object in subfolder_object:
                        articles_path.append(file_object.path) 
            
        
    
        pipeline = PmcRetrievalPipeline(
            articles_path=articles_path, #[1:100000], #for quick testing cut papers down here
            master_keyword=args.master_keyword,
            textsearch_keywords=args.textsearch_keywords.split(","),
            master_keyword_rag=args.master_keyword_rag,
            keywords_for_rag=args.keywords_for_rag.split(","),
            threshold_embedding=args.threshold_embedding,
            llm_keywords_for_approval=llm_keywords_for_approval,
            llm_approvals=int(args.llm_approvals),
            run_embedding_in_multithreaded_batches=args.run_embedding_in_multithreaded_batches,
            number_of_batches_to_run_embedding=args.number_of_batches_to_run_embedding,
            suffix_for_saved_docs=args.suffix_for_saved_docs,
        )
        print("Keyword Search - Looking for relevant papers:")
        
        try:
            pipeline.summarize_papers()
        
        except Exception as e:
            print("pipeline failed to finish - check errors")
            tb = traceback.format_exc()
            raise RuntimeError(f"pipeline failed to finish:\n{e}\n{tb}") from e 
        
        finally:
            cache_class(object_name=CACHE_FILE_NAME, object=pipeline, save=True)
    


if __name__ == "__main__":
    
    LOG_DIR = Path("../../log")
    LOG_DIR.mkdir(exist_ok=True)
    CACHE_DIR=Path("../../cache")
    CACHE_DIR.mkdir(exist_ok=True)

    FLAG_FILE: Path = LOG_DIR/"flag.txt"
    ERROR_LOG: Path =LOG_DIR/"error_log.txt"
    ARGUMENTS_FILE: Path =LOG_DIR/"last_arguments.txt"
    PROCESS_LOG: Path=LOG_DIR/"process.log"
    CURRENT_FILE_NAME: Path=LOG_DIR/"current_file_name.txt"

    CACHE_FILE_NAME: str=read_CACHE_FILE_NAME()
    
    pipeline: PmcRetrievalPipeline | None
    pipeline=None

    log: Any = PROCESS_LOG.open("w", encoding="utf-8", buffering=1)

    sys.stdout = log
    sys.stderr = log
    #Start watchdog thread to kill process if flag returns false:
    t = threading.Thread(target=watchdog, args=(FLAG_FILE,), daemon=True) 
    t.start()

    if read_flag(FLAG_FILE):
        try:
           
            main()
            print("\n###FINISHED SUCCESSFULLY####")
        except Exception as e: 
            with open(ERROR_LOG, "a", encoding="utf-8") as f: 
                f.write(f"Exception occurred:\n{type(e).__name__}: {e}\n\n") 
                print("An error occurred. Details were written to error_log.txt")

        finally:
            write_flag(False,FLAG_FILE)
    else:
        print("Python job already running")
