# Tested with Python 3.14.1
# run with: streamlit run app.py --server.port=8501

import os
import sys
from typing import Any
import subprocess
import time
from pathlib import Path
from typing import overload, Literal


import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import streamlit.components.v1 as components



def download_button(result_name, word_report=True) -> None:
        """Creates a download button for the word report after generation or download the arguments.txt after the first run"""
        if word_report:
            if os.path.exists(RESULT_DIR/result_name) and result_name!="": 
                with open(RESULT_DIR/result_name, "rb") as f: 
                    data = f.read()
                    st.text('File ready to download')
                    st.download_button( label="📝Download Word Report", data=data, file_name=os.path.basename(RESULT_DIR/result_name), mime="application/octet-stream", )
            else: 
                st.warning("Result file not found.")
        else:   
            try:
                with open(result_name, "rb") as f: 
                    data: bytes = f.read()
                    #for the arguments variable are already written with a path to the file
                    st.download_button( label="Download Last Arguments", data=data, file_name=os.path.basename(result_name), mime="application/octet-stream",
                    help=
                    """
                    This will download a text file containing the last set of arguments in case you want to copy paste some for a new run
                    """) 
            except:
                st.warning("Result file not found. Run your first run ")


def search_file_form(session_value) -> tuple[str | None, bool]:
    """function to create a search button and form to look up filenames to download"""
    
    with st.form(key="file_download_text"):
        download_text_file_name_: str | None=st.text_input(
            "File Name to download", 
            value=session_value)
            
        submit_download_: bool=st.form_submit_button("Search File")  
    return download_text_file_name_, submit_download_

def read_flag() -> bool: 
    """reads from flag txt and returns True or False for logging states"""
    if not os.path.exists(FLAG_FILE): 
        return False 
    with open(FLAG_FILE, "r") as f: 
        return f.read().strip() == "True" 

def write_flag(value: bool) -> None:
    """reads from flag txt and returns True or False for logging states"""
    with open(FLAG_FILE, "w") as f: 
        f.write("True" if value else "False")

def write_current_file_name(value: str) : 
    """Writes the last generated word document filename into a text file"""
    with open(CURRENT_FILE_NAME, "w") as f: 
        f.write(value if value else "")

def read_current_file_name() -> str:
    """Reads the last generated file name""" 
    if not os.path.exists(CURRENT_FILE_NAME): 
        return "" 
    with open(CURRENT_FILE_NAME, "r") as f: 
        return f.read().strip()

def write_start_time()  -> None:
    """Logs the start time to a text file""" 
    with open(TIME_SHEET, "w") as f: 
        value: float=time.time()
        f.write(str(value))

def read_start_time()  -> float:
    """Reads start time from a text file"""
    if not os.path.exists(TIME_SHEET): 
        return 0 
    with open(TIME_SHEET, "r") as f: 
        return float(f.read().strip())

def read_output_index() -> int:
    """Reads current output lines from a text file"""
    if not os.path.exists(OUTPUT_INDEX_FILE): 
        return 0 
    with open(OUTPUT_INDEX_FILE, "r") as f: 
        return int(f.read().strip())

def write_output_index(current_line)  -> None:
    """"writes current output lines from a text file""" 
    last_line=read_output_index()
    
    if last_line!=current_line:
        with open(TIME_SHEET, "w") as f: 
            f.write(str(current_line))

@overload 
def read_process_log(line_index: int, return_line_number: Literal[False]) -> str: ...
@overload 
def read_process_log(line_index: int, return_line_number: Literal[True]) -> int: ...

def read_process_log(line_index: int, return_line_number: bool=True)  -> str|int:
    """
    retrieves a str line from the process log file or the total line numbers.
    """
    try: 
        with open(PROCESS_LOG_FILE, "r", encoding="utf-8") as f: 
            lines: list[str] = f.readlines()
            total_n_lines=len(lines)

        if return_line_number:
            return total_n_lines
        else:
            try: 

                return lines[line_index]
            except:
                return ""
    
    except FileNotFoundError: 
        if return_line_number:
            return 0
        else:
            return "" 
def read_all_process_log() ->str:
    """
    reads and returns the entire log
    """
    try: 
        with open(PROCESS_LOG_FILE, "r", encoding="utf-8") as f: 
            lines: list[str] = f.readlines()

        return "\n".join(line.rstrip("\n") for line in lines)

    
    except: 
            return "" 


def clean_process_file() -> None:
    """Empties output log file"""
    with open(PROCESS_LOG_FILE,"w", encoding="utf-8", buffering=1) as file:
        file.write("") 

def running_animation(value)-> str:
    """Returns a clock emojy to use in a loop to generates a running clock animation"""
    try:
        symbol=['🕛','🕐','🕑','🕒','🕓','🕔','🕕','🕖','🕗','🕘','🕙','🕚'][value]
    except:
        symbol=""
    return symbol
 
def autoscroll() -> None:
        """create an autoscrol html component to use in the text output area"""
        components.html(
            """
            <script>
            var elem = window.parent.document.querySelector('.stTextArea textarea');
            if (elem) { elem.scrollTop = elem.scrollHeight; }
            </script>
            """,
            height=0,
        )

def main() -> None:
    ### Streamlit UI###
    #page layout:
    st.set_page_config(
        page_title="KELLMA search",
        page_icon="🧿", #🔎
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("### KELLMA Search\n*Keyword embedded large language model approved search*")
    #create a running log thoughout execution:
    if "process" not in st.session_state:
        st.session_state.process = None

    if "time_start" not in st.session_state:
        st.session_state.time_start=read_start_time()
    
    if "time_stamp" not in st.session_state:
        st.session_state.time_stamp=0

    if "log_text" not in st.session_state:
        st.session_state.log_text = {"output_history":"", "new_line":""}

    if "download_file_name" not in st.session_state:
        st.session_state.download_file_name = ""
    if "download_available" not in st.session_state:
        st.session_state.download_available=False
    if "existing_job_running" not in st.session_state:
        st.session_state.existing_job_running=read_flag()
    else:
        st.session_state.existing_job_running=read_flag()
    
    if "output_line_index" not in st.session_state:
        st.session_state.output_line_index = read_output_index()

    if "while_loop_exit_success" not in st.session_state and not st.session_state.existing_job_running:
        st.session_state.while_loop_exit_success=False
        clean_process_file()
    elif "while_loop_exit_success" not in st.session_state:
        st.session_state.while_loop_exit_success=False

    if "while_loop_int" not in st.session_state:
        st.session_state.while_loop_int=0


    with st.sidebar:
        st.sidebar.markdown("## *Current/last file name:*")
        st.text(read_current_file_name())
        st.sidebar.markdown("## Advanced Options :")
        threshold_embedding: float = st.number_input("Threshold score (RAG): ", value=0.46, help="""
                                                     This threshold cuts off the number of text chunks to send to the llms for approval
                                                     - the higher the number, the more 'meaningful' the text chuncks are send to the llm and low scoreing text is discarded. 
                                                     In other words the higher the faster and more accurate, but you may end up with no references or lose out on some.
                                                     A very low score will feed way too much into LLM approvals which very heavy computationally and will make the summary very inaccurate""")
        
        widget_cache_enabled: bool=st.radio("Use cache to try another RAG threshold?: ", [True,False], index=0, help="""RAG and paper loading steps will be skipped 
                                            if the pipeline was already run with the same 'Master Keyword' and namesuffix:""")
        
        llm_approvals: int = st.number_input("Number of LLM approvals: ", value=2, help="""
                                             Currently 4 llms are designed to read the text chuncks and approve of them, 
                                             this values decides how many llms need to approve in total for it to be kept for the final document and summary.""")
        
        run_embedding_in_multithreaded_batches: bool=st.radio("Run in multithreadded batch mode (Recommended): ", [True,False], index=0, help="""
        This decides wheter the text chunks are split into multiple threads where per thread the RAG semantic keywords are send one by one to the RAG algorithm per text chunk
        or if the papers are processed sequenctially but the keywords are split up into multiple threads and send to the RAG algorithm simoultaniously. 
         - Recommended to leave True, unless you use very little RAG keywords in case setting False can speed up RAG
        """)
        
        number_of_batches_to_run_embedding: int = st.number_input("Number of batches per thread (default= 10): ", value=10,
            help="""
            This number is used to split the list of multiple papers/text chunks into smaller lists that are processed in parallel.
            For 10, one thread will process 10 elements sequentially
             - Recommended to leave at 10  
            """)
        
        suffix_for_saved_docs: str=st.text_input("Namesuffix for saved documents: ", value="st", help="""
        This suffix will be used to save the final report and to save the cache file for faster reruns
        """)
        path_to_pmc_xmls: str= st.text_input("Path to papers (pmcxxx.xml): ", value="../../PMC", help=
        """
        The path to the location where the folder stores all of the pmc xml files
        """)
        multiple_folders: bool= st.radio("Articles in multiple subfolders?: ", [True,False], index=0, help="""
        Do you have your articles stored in multiple subfolders (max depth 1) eg PMC_1 PMC_2 -> True
        If all are in one folder: False
        """)
        st.space()
        download_button(ARGUMENTS_FILE, word_report=False)



    with st.form(key="gene_query_form"):

        master_keyword = st.text_input("Master Keyword: ", value="tbr1", help=
        """
        Full article texts is split into chunks of 200 words. 
        This keyword is used to fetch out all(!) of the text chunks containing this keyword.
        """)
        textsearch_keywords = st.text_input("Textsearch Keywords: ", value="neuron, Brain, Cortex", help=
        """
        These keywords are used in cojunction with the 'Master Keyword', 
        for a given text chunk, if the master keyword is present, and one of these keywords as well, 
        the entire article will be retained for further processing
        """)
        
        master_keyword_rag = st.text_input("Master Keyword (RAG): ", value="tbr1 gene",
        help=
        """
        This is used to boost the RAG algorithm score on text chuncks that contain the true master keyword you were looking for.
         - eg. 'sox1' as 'Master keyword (first input)' will fetch sox1 sox10 sox11 etc and any other words containing "sox1" mid word.
        By using a similar RAG master keyword eg. "Sox1 Gene expression" it will likely boost the semantic score of text chunks where exactly "sox1 gene" is present.
        """)
        keywords_for_rag = st.text_input("Semantic keywords (RAG): ", value="neuron,cortex,brain,marker for neurons, neural precursors,progenitor,neuronal lineage marker,expression",
        help=
        """
        The RAG algorithm will used these keywords individually to see wheter the whole text chunk is semantically similar to them and give one score per keywords.
         - All of the scores are averaged together at the end, meaning if the keywords are semantically far away (eg. desk, desk), then the overall final score will be lower
         and result in no references being feed into the next step if the RAG threshold is set too high.

        """)
        
        llm_keywords_for_approval_element1 = st.text_input("LLM keywords: ", value="Neurons - Markers for neurons - Brain structures",
        help=
        """
        This input is used to construct the prompt for the LLMs that will 'read' and 'approve' them. This should be like a hashtag that describe what the text you look for should be about.
        """)
        llm_keywords_for_approval_element2 = st.text_input("LLM context: ", value="cell type expression, genes related or involved to neuronal processes, neuronal processes (even if broader), brain structures, or any biological process closely related",
        help=
        """
        This will also be used to build the prompt and should be more about giving context to the LLM to understand what exactly is important to look for. 
        This can be keywords, but also whole sentences
        """)
        
        
       
        submit = st.form_submit_button("Run Pipeline")
    
    
    

    output_height: int=400
    # Placeholder for terminal output
    st.write("Output")
    
    #if history log is not empty autoscroll to the bottom
    if not st.session_state.existing_job_running and st.session_state.log_text["output_history"]!="": 
        terminal: DeltaGenerator = st.empty()
        terminal.text_area(label="hidden", label_visibility="hidden", value=st.session_state.log_text["output_history"],height=output_height)
        autoscroll()
    #if empty simply show
    elif not st.session_state.existing_job_running:
        terminal: DeltaGenerator = st.container(height=output_height, vertical_alignment ="bottom")
        
    else:
        terminal: DeltaGenerator = st.container(height=output_height, vertical_alignment ="bottom")
    
    st.session_state.cancel = st.button("Kill Ongoing Run", help="""
    This will kill any ongoing run, or if none is running reset/clean the webpage ui""")
    
    #this part creates the search file form is starting, the name will be empty
    #when running the code, the session will keep the previous download and show the name
    #if the user tries to look up a name, the session overwrites it to the user file name
    
    ###DOWNLOAD FORM####
    download_file_name: str |None
    submit_search: bool | None
    placeholder: str |None

    submit_search=None
    download_file_name=None

    if st.session_state.download_available==False:
        
        download_file_name,submit_search=search_file_form(st.session_state.download_file_name)


    if st.session_state.download_available==True:
        placeholder,submit_search=search_file_form(st.session_state.download_file_name)
        download_button(result_name=st.session_state.download_file_name)
        st.session_state.download_available=False


    elif st.session_state.download_available=="user try":
        download_file_name,submit_search=search_file_form(st.session_state.download_file_name)
        st.session_state.download_file_name=download_file_name
        download_button(result_name=download_file_name)
    
    if submit_search:
        st.session_state.download_available="user try"
        st.session_state.download_file_name=download_file_name
        st.rerun()

    ####JOB TIME Widget####
    job_state: DeltaGenerator=st.empty()
    if not read_flag():
        job_state.write("🟢 Ready")
    else:
        job_state.empty()
    #this submits the run pipeline button and form for the main function call
    
    if submit and st.session_state.existing_job_running==False:
        write_start_time()
        st.session_state.time_start=read_start_time()
        
        write_flag(True)
        st.session_state.existing_job_running=True
        st.session_state.download_available=False
        file_name_submitted: str=f"llm_rag_{master_keyword}_llma_{llm_approvals}_{suffix_for_saved_docs}.docx"
        write_current_file_name(file_name_submitted)
        
        # Building args from webui values
        args: dict[str, Any] = {
            "articles_path": path_to_pmc_xmls,
            "master_keyword": master_keyword,
            "textsearch_keywords": [no_blanks for no_blanks in (txt.strip() for txt in textsearch_keywords.split(",")) if no_blanks],
            "master_keyword_rag": master_keyword_rag,
            "keywords_for_rag": [no_blanks for no_blanks in (txt.strip() for txt in keywords_for_rag.split(",")) if no_blanks],
            "threshold_embedding": threshold_embedding,
            "llm_keywords_for_approval_e1": llm_keywords_for_approval_element1.strip(),
            "llm_keywords_for_approval_e2": llm_keywords_for_approval_element2.strip(),
            "llm_approvals": llm_approvals,
            "run_embedding_in_multithreaded_batches": run_embedding_in_multithreaded_batches,
            "number_of_batches_to_run_embedding": number_of_batches_to_run_embedding,
            "suffix_for_saved_docs": suffix_for_saved_docs,
            "cache_enabled":str(widget_cache_enabled),
            "multiple_folders":str(multiple_folders),


        }


        # Building command to use sys.executable
        cmd: list[Any] = [
            sys.executable,
            "-X", "utf8", 
            "-u", "pipeline_init.py",
            "--articles_path", args["articles_path"],
            "--master_keyword", args["master_keyword"],
            "--textsearch_keywords", ",".join(args["textsearch_keywords"]),
            "--master_keyword_rag", args["master_keyword_rag"],
            "--keywords_for_rag", ",".join(args["keywords_for_rag"]),
            "--threshold_embedding", str(args["threshold_embedding"]),
            "--llm_keywords_for_approval_e1", str(args["llm_keywords_for_approval_e1"]),
            "--llm_keywords_for_approval_e2", str(args["llm_keywords_for_approval_e2"]),
            "--llm_approvals", str(args["llm_approvals"]),
            "--run_embedding_in_multithreaded_batches", str(args["run_embedding_in_multithreaded_batches"]),
            "--number_of_batches_to_run_embedding", str(args["number_of_batches_to_run_embedding"]),
            "--suffix_for_saved_docs", args["suffix_for_saved_docs"],
            "--cache_enabled",args["cache_enabled"],
            "--multiple_folders",args["multiple_folders"],
        ]

        # Start subprocess and stream stdout
        st.session_state.process = subprocess.Popen(
            cmd,
            shell=False,
            cwd = Path(__file__).resolve().parent / "backend", # sets CWD to KELLMA/src/backend
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=1,
            encoding="utf-8",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        st.session_state.output_line_index = 0
        #after init wait a bit for the process to start:
        time.sleep(0.5)
        st.rerun()

    elif submit:
            st.rerun()

    ######While Loop to update text fields while running####
    while read_flag()==True:
        
        ####OUTPUT READ PRINT SECTION#####
        #read file, and then scroll automatically down and append new last file line to first line's output 
        new_text: str
        total_log_lines: int
        total_log_lines= read_process_log(st.session_state.output_line_index, return_line_number=True)
        #to give a boost if the user refreshes:
        if total_log_lines-st.session_state.output_line_index>100:
           st.session_state.output_line_index+=99

        if st.session_state.output_line_index<total_log_lines:
            
            new_text=read_process_log(st.session_state.output_line_index, return_line_number=False)
            st.session_state.output_line_index+=1
        else:
            new_text=st.session_state.log_text["new_line"]


        if st.session_state.cancel:
        # write the kill flag, and update terminal quickly
                st.session_state.output_line_index=0
                write_flag(False)
    
        if st.session_state.log_text["new_line"]!= new_text:
            st.session_state.log_text["new_line"] =  new_text
            terminal.text(st.session_state.log_text["new_line"])
        
        ###TIME ANIMATION####
        elapsed: float | Any = time.time() - st.session_state.time_start
        total_seconds = int(elapsed)
        hours: int = total_seconds // 3600
        minutes: int = (total_seconds % 3600) // 60
        seconds: int = total_seconds % 60
        st.session_state.time_stamp=f'{hours}:{minutes:02d}:{seconds:02d}'
    
        #Display time:
        job_state.text(f"🔴 Running... {st.session_state.time_stamp} {running_animation(st.session_state.while_loop_int)}")
        if st.session_state.while_loop_int>10:
            st.session_state.while_loop_int=0
        else:
            st.session_state.while_loop_int+=1
        
        ###EXIT CONDITIONS#####
        #flag exit of loop and handle cancel button
        st.session_state.while_loop_exit_success=True
        if st.session_state.cancel: 
            st.session_state.cancel=False
            st.session_state.log_text["output_history"]=f'{read_all_process_log()}\n####Process killed by user####'    
            st.session_state.download_available=False
            st.session_state.while_loop_exit_success=False
            st.session_state.output_line_index=0
            #wait for flag to be writen and process to be killed before rewriting the log the file
            time.sleep(8)
            clean_process_file()
            st.rerun()
        #wait to avoid cpu spike
        time.sleep(0.1)

    #if killed wait a bit to reinit page, reinit session state cancel 
    if st.session_state.cancel:
       
        st.session_state.cancel=False
        for key in st.session_state.log_text:
            st.session_state.log_text[key]=""
        clean_process_file()
        st.session_state.download_file_name=""
        st.session_state.download_available=False
        st.session_state.output_line_index=0
        st.rerun()

    #when exiting while loop (end of process) init page and add filename to download
    if read_flag()==False and st.session_state.while_loop_exit_success:
    
        st.session_state.cancel=False
        st.session_state.download_available=True
        st.session_state.while_loop_exit_success=False
        st.session_state.download_file_name=read_current_file_name()
        st.session_state.log_text["output_history"]=f'{read_all_process_log()}'
        #clean_process_file() #Remove this in case you want to keep the log after a run
        st.rerun()        

    if read_flag()==False:
        st.session_state.download_file_name=""

if __name__=="__main__":
    
    #Log files:
    LOG_DIR = Path("../log")
    LOG_DIR.mkdir(exist_ok=True)

    FLAG_FILE: Path = LOG_DIR/"flag.txt"
    CURRENT_FILE_NAME :Path = LOG_DIR/"current_file_name.txt"
    TIME_SHEET :Path =LOG_DIR/"process_start_time.txt"
    OUTPUT_INDEX_FILE: Path=LOG_DIR/"output_index.txt"
    ARGUMENTS_FILE :Path=LOG_DIR/"last_arguments.txt"
    PROCESS_LOG_FILE :Path =LOG_DIR/"process.log"
    
    RESULT_DIR = Path("../results")
    RESULT_DIR.mkdir(exist_ok=True)
    
    main()