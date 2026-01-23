import json
from typing import Any
from pathlib import Path

def prase_pmc_article(textsearch_keywords, master_keyword, file_location:str|None=None) -> list[str]:
    """Reads a JSON file containing PMC articles and returns a list of strings representing the PMC id, title and text."""

    if file_location is None:
        raise ValueError("File location cannot be None")
    
    # Step 1: Load JSON from a file
    try:
        with open(file_location, "r") as f:
            data: Any = json.load(f)
            data: Any = data["documents"][0]
    except:
        # Append the filename + error message to a text file
        #Log files:
        log_dir: Path = Path("../../log")
        log_dir.mkdir(exist_ok=True)

        with open(log_dir/"failed_files.txt", "a", encoding="utf-8", buffering=1) as log: 
            log.write(f"Check file - Error with json loading: {file_location} \n")
        data: Any ={}

    # Step 2: See all top-level keys
    #Get ID and Title:
    try:   
        id_title_article: tuple[str,str]=[(key["infons"]["article-id_pmc"], key["text"]) for key in data["passages"] if (key["infons"]["section_type"] in ["TITLE"])][0]
        Article_text: str=(" ".join([(key["text"]) for key in data["passages"] if (key["infons"]["section_type"] in ["TITLE","ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL"])]))
        Article_values: list[str]=[id_title_article[0],id_title_article[1],Article_text]
    except:
        Article_values=["","",""] #first index; paper ref, second title, third article text
    #Now select based on masterkeyword and any other words
    if master_keyword.lower() in Article_values[2].lower() and any(word.lower() in Article_values[2].lower() for word in textsearch_keywords):
        return Article_values
    else:
        return ["","",""]


def split_into_words_for_retrieval(master_keyword, parsed_article, chunck_length: int=200, sliding_window: int|None=6) -> list[str]:
    """This function splits article text into smaller chuncks of text
    Args:
        parsed_article (list[str]): A list containing the article ID, title and text.
        chunck_length (int): The length of each chunk.
        sliding_window (int|None): Controls the stepsize in the loop, if eg 100 loops every 100 words and adds the initial index+chuncklength, creating a sliding window. If None, it uses the chunck length to create no overlapps.
    """
    if parsed_article[0]=="":
        chunks = [""]
    else:
        words: list[str] = parsed_article[2].split()
        if sliding_window:
            chunks: list[str] = [" ".join(words[i:i+chunck_length]) for i in range(0, len(words), chunck_length-sliding_window)]

        else:
            chunks: list[str] = [" ".join(words[i:i+chunck_length]) for i in range(0, len(words), chunck_length)]
    
    #restrict text to bigger than 3 chunks eg 300 words
    if len(chunks) <3:
        chunks = [""]

    #only keep chunks with master keyword:
    chunks=[item if master_keyword.lower() in item.lower() else "" for item in chunks]
    #clean empty chuncks
    chunks = [item if item != "" else item for item in chunks if item.strip()]
    #remouve empty:
    if chunks ==[]:
        chunks=[""]

    return(chunks)

def return_single_pmc_article_processed(textsearch_keywords, master_keyword, article_path) -> dict[str,list[str]]:
    """This function gets the article id, and a list of chuncks text and returns a dictionnary ID:Parsed text. Remouves empty text or missing articles"""
    new_article_list: dict[str,list[str]]={"":[]}
    parsed_article: list[str]=prase_pmc_article(textsearch_keywords, master_keyword, article_path)
    chunked_text: list[str]=split_into_words_for_retrieval(master_keyword, parsed_article)
    #remouve empty chunks entries
    if chunked_text[0] =="":
        new_article_list={"":[""]}
    else:
        new_article_list={parsed_article[0]:chunked_text}
    return(new_article_list)