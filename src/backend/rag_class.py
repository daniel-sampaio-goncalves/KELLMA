#Tested with Python 3.14.1 and Ollama v0.15.1

import re
from typing import Optional, Any, Union, Sequence
import numpy as np
import numpy.typing as npt 
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


from ollama import generate, GenerateResponse, embed
from sklearn.metrics.pairwise import cosine_similarity

class Refined_Semantic_matching:
    """
    This Class is designed to refine the semantic similarity scores by averaging them.
    It takes in a list of words to check for semantic similarity and a list of vectors to compare against. 
    It then calculates the average semantic similarity score for each word using the provided vectors.
    It uses pythons concurrent.futures.concurrent.futures.ThreadPoolExecutor to send multiple words to Ollama's qwen3-embedding:latest embedding model in parallel.
    Finally, it uses a LLM to refine the top candidates.
    Attributes:
       array_word_to_check_for_embeddings (list): A list of words to check for semantic similarity.
       vector_items_to_compare (list): A list of vectors to compare against.
       checking_prompt_for_LLMs (list): A list of two elements; the first is a word semantic or sentencence you're looking to identify, sencod: a description for better context to the llms eg: ["Neuronal Markers","cell type expression, genes related or involved to neuronal processes, neuronal processes (even if broader), brain structures, or any biological process closely related"]

    """
          
    def __init__(
            self, 
            array_word_to_check_for_embeddings: list[str]=[], 
            vector_items_to_compare: list[str]=[], 
            checking_prompt_for_LLMs:list[str]=["",""],
            adjust_for_rank: bool = True,
            embedding_rank_percent_cutoff: float= 0.6,
            embedding_cpu_threadding: bool = False
            ) -> None:
        
        self.array_word_to_check_for_embeddings: list[str] = array_word_to_check_for_embeddings #self.array_word_to_check_for_embeddings
        self.vector_items_to_compare: list[str] = vector_items_to_compare
        self.checking_prompt_for_LLMs: list[str]=checking_prompt_for_LLMs
        
        self.similarity_results: list[list[tuple[str, float]]]= [] 
        self.similarity_results_adjusted__for_rank: list[list[tuple[str, float]]] = []
        #self.averaged_similarity_results: Union[list[tuple[str, float]],None] = None
        #self.averaged_similarity_results_adjuste_for_rank: Union[list[tuple[str, float]],None] = None
        self.averaged_similarity_results: list[tuple[str, float]] = []
        self.averaged_similarity_results_adjuste_for_rank: list[tuple[str, float]] = []
        self.raw_embedding_result:list[list[tuple[str, float]]] =[]
        
        self.prompt_to_inject: str =f"<INSTRUCT>: We are looking specifically for any words or sentences related to '{self.checking_prompt_for_LLMs[0]}'. Note that it has to be closely related and sometimes may contain typos. Eg.specifc {self.checking_prompt_for_LLMs[1]}\n\n<Answer>: "

        self.adjust_for_rank: bool = adjust_for_rank
        self.embedding_rank_percent_cutoff: float= embedding_rank_percent_cutoff
        self.embedding_results_cut_off: Union[list[tuple[str, float]],None] = None
        self.vector_items_to_compare_cut_by_rank: list[str]=[]
        self.embedding_cpu_threadding: bool = embedding_cpu_threadding

        self.model_list: list[str]=["gpt-oss:20b","qwen3:32b", "gemma3:27b", "deepseek-r1:32b"]
        self.LLM_Output_List: list[tuple[str,dict[str,int]]]=[]
        self.LLM_averaged_results: list[dict[str,int]]= []          
        self.LLM_Approved_keyword_list: list[str]=[]
        self.LLM_last_AI_response: str=''
        #filter any empty strings from the embedding keylist to avoid empty strings in embedding results:
        self.array_word_to_check_for_embeddings = self._clean_attribute_list(self.array_word_to_check_for_embeddings)
        self.vector_items_to_compare = self._clean_attribute_list(self.vector_items_to_compare)
        
    def _clean_attribute_list(self, attrribute_name: list[str]) -> list[str]:
        '''This function removes any empty strings from a list of a list[str].'''
        return [word for word in attrribute_name if word.strip()]

    def cosine_similarity_create_ranks (self,word_to_check_: str) -> list[tuple[str, float]] :
        """
        This function calculates the cosine similarity between a given word and all words in given list.
        It returns a list of tuples, where each tuple contains a word and its corresponding cosine similarity score.
        """
        #Call qwen3-embedding:latest

        word_to_check_embedding: npt.NDArray[Any] = np.array(
            embed(
                model= "qwen3-embedding:latest",
                input =word_to_check_,
                options={"num_ctx": 2084}
                ).embeddings)
        cosine_vector_list: list[npt.NDArray[Any]]  = []
        
        #Get cosine similarity for word to check and compare to the enitre list: 
        for word_to_check_vector in self.vector_items_to_compare:
            embedding_vector: npt.NDArray[Any] = np.array(embed(
                model= "qwen3-embedding:latest",
                input =word_to_check_vector,
                options={"num_ctx": 2084}
                ).embeddings)
            comparison_cosine_: npt.NDArray[Any] = cosine_similarity(word_to_check_embedding, embedding_vector)
            cosine_vector_list.append(comparison_cosine_)


        #add the words/sentence back:
        Word_Pair_Embedding_result_: list[tuple[str, float]] = list(zip(self.vector_items_to_compare, [float(cosine_value[0][0]) for cosine_value in cosine_vector_list]))
        #sort highest to lowest:
        return Word_Pair_Embedding_result_
    
    
    def multi_check_embedding_similarity_parallel(self)-> list[list[tuple[str, float]]] :
        """This function takes a list words to check and a list of vectors, then compares the word's embedding with each vector using cosine similarity.
        It returns a list of tuples containing the words and their corresponding cosine similarities. Uses Threadpoolexcecutor to speed up the process. (10x)"""
        final_list_: list[list[tuple[str, float]]] =[]
        
        if self.embedding_cpu_threadding:
            max_workers_cpu:int = multiprocessing.cpu_count()
     
        else:
            max_workers_cpu:int =1

        #spawning multiple processors to use several ollama requests speed up the process (GPU went from 20% sequentially to 100%)
        #for a same sequential 1min 30sec function to 16 secs
        with ThreadPoolExecutor(max_workers=max_workers_cpu) as executor:
            final_list_ = list(
            executor.map(self.cosine_similarity_create_ranks, self.array_word_to_check_for_embeddings)) # array_word_to_check_for_embeddings this is the list to add one by one
        self.raw_embedding_result=final_list_

        return final_list_
    

    def factor_ranking_to_similarity_score(self, similarity_list:  list[list[tuple[str, float]]])-> list[list[tuple[str, float]]] :
        """This functions adjusts the similarity score for rankng of each element in the different lists.
        The ranking is based on the position of each element compared to the entirety of the tested words.
        Ranking is normalized between 0 and 1 and then averaged with the previously calculated similarity score."""
        ranked_list_:list[list[tuple[str, float]]] = [sorted(element_, key=lambda x: x[1], reverse=False) for element_ in similarity_list]

        Score_adjusted_list:list[list[tuple[str, float]]]  = [
            list(
                (label, ((rank / len(sublist)) + float(score)) / 2.0)
                for rank, (label, score) in enumerate(sublist, start=1)
            )
            for sublist in ranked_list_
        ]

        return Score_adjusted_list

    def average_similarity(self, list_of_tuples: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
        """This function takes a list of tuples containing words and their corresponding cosine similarities, then calculates the average similarity."""
        #sort the nested list to get same order for each list:
        list_of_tuples=[sorted(element_, key=lambda x: x[0], reverse=False) for element_ in list_of_tuples]
        
        how_many_words_: range = range(len(list_of_tuples[0]))
        list_of_average_similarity: list[float] = []
        list_of_words_checked: list[str] = []

        list_of_words_checked: list[str] = [list_of_words_checked[0] for list_of_words_checked in list_of_tuples[0]]
        
        for word_index in how_many_words_:
            average_similarity: float = float(np.mean([sublist[word_index][1] for sublist in list_of_tuples]))
            list_of_average_similarity.append(average_similarity)
            
        Final_return_average_similarity_list: list[tuple[str, float]] = list(zip(list_of_words_checked, list_of_average_similarity))
        Final_return_average_similarity_list= sorted(Final_return_average_similarity_list, key=lambda x: x[1], reverse=True)
        return Final_return_average_similarity_list
    
    def run_embedding_scores(self) -> None:
        """This function that calls class methods to perform the similarity check and average the results."""
        self.similarity_results= self.multi_check_embedding_similarity_parallel()
        
        self.similarity_results_adjusted__for_rank=self.factor_ranking_to_similarity_score(self.similarity_results)
        
        #also arrange for a nicer output
        self.similarity_results = [sorted(element_, key=lambda x: x[0], reverse=False) for element_ in self.similarity_results]
        self.similarity_results_adjusted__for_rank = [sorted(element_, key=lambda x: x[0], reverse=False) for element_ in self.similarity_results_adjusted__for_rank]
        self.array_word_to_check_for_embeddings = sorted(self.array_word_to_check_for_embeddings, key=lambda x: x, reverse=False)
        self.vector_items_to_compare = sorted(self.vector_items_to_compare, key=lambda x: x, reverse=False)

        self.averaged_similarity_results_adjuste_for_rank = self.average_similarity(self.similarity_results_adjusted__for_rank)
        self.averaged_similarity_results = self.average_similarity(self.similarity_results)
    
####Second Part: Take Ranking from embedding, do a firs cut off, and feed it to LLMs to double check for correctnes and return final result dictionnary 
    def cut_off_embeding_list_by_rank(self) -> None:
        '''This function takes the ranking from embedding and does a first cut off based on the rank percentage cutoff before feeding it to LLMs for refinment'''
        if self.adjust_for_rank:
            primary_list: list[tuple[str, float]]| None = self.averaged_similarity_results_adjuste_for_rank
        else:
            primary_list: list[tuple[str, float]] | None = self.averaged_similarity_results
        
        if primary_list is None:
           primary_list = []

        primary_list=sorted(primary_list, key=lambda x: x[1], reverse=True)
        self.embedding_results_cut_off=[(keyword, score) for (keyword, score) in primary_list if score >= self.embedding_rank_percent_cutoff]
        self.vector_items_to_compare_cut_by_rank= [words[0] for words in self.embedding_results_cut_off]

    def summarize_result_matrix(self, Output_LLM_dict_list) -> list[dict[str, int]]:
        '''This function takes the result from Check_LLM_Semantics that returns a list of results per model used [model,{keyword: result}] and summarises the results into a single list of dictionnaries {keyword: average_result}'''
        Matrix_of_keyword_results: list[list[int]] =(
            [
            [keyword[keys_word] for keys_word in self.vector_items_to_compare_cut_by_rank ]  #iterate over the words to check and retrieve a matrix of results
            for keyword in                          #for each result dict
                (marker_dict[1] for marker_dict in # iterate over the marker_dict list and get a list of the marker_dict results
                    
                    (Output_LLM_dict_list[index_model[0]] for index_model in enumerate(Output_LLM_dict_list)) # enumerate the models and get a list of the models results unpack models
                )
            ]
        )

        summed_Matrix: list[int] =[sum(col) for col in zip(*Matrix_of_keyword_results)] # *unpacks the matrix of keyword results and sum each column, zip groups collumns together in a tuple
    
        return [{words_in_keywords: results_from_matrix} for words_in_keywords, results_from_matrix in zip(self.vector_items_to_compare_cut_by_rank, summed_Matrix)]


    def get_response_generate(
            self, 
            prompt_: str, 
            model: str,  
            reasoning_effort: Optional[str] = "high", 
            system_prompt_: str ="", context_: Sequence[int]|None=None,
            context_tokens: int =2084,
            stop_tokens: int|None=5000
            ) -> GenerateResponse:
        '''
        Function that talks to ollama api using both a system prompt anand returns a response.
        '''
        if model=="gpt-oss:20b":
            allowed = {"low", "medium", "high"}
            if reasoning_effort not in allowed:
                raise ValueError(f"reasoning-effort must be one of {allowed}")
            extra_body_=reasoning_effort
        else:
            extra_body_=None
            
        response_: Any = generate(
            model=model,
            prompt=prompt_,
            stream=False,
            think= extra_body_, # type: ignore      # "low" | "medium" | "high" for chatpgt differs from bool
            system=system_prompt_,
            context=context_,
            options={
                "seed": 42,
                "temperature": 0.1,
                "num_ctx": context_tokens,
                "top_k": 10,
                "top_p": 0.9,
                "num_predict":stop_tokens,
                "repeat_penalty":1
                
            }
        )       
                #Some explanations of model parameters:
                
                #"stop":["yes","Yes","YES"]
                #"num_ctx": 16384, #context length of the model
                #"temperature": 0.1, #values (e.g., 0.0–0.2) → deterministic, repetitive outputs. Higher values (e.g., 0.7–1.0) → more creative, varied outputs.
                #"top_k": 10, #Limits sampling to the top k probable tokens. 1= always pick the highest probability token. 5 = 5 highest tokens
                #"top_p": 0.9, #Chooses from tokens whose cumulative probability ≤ p. top_p=1.0 → no restriction. Lower values reduce randomness.use only tokes with combined prob higher than set eg 0.5 for 50%
                #"num_predict":5000 Number of tokens before the ollama stops
                #"repeat_penalty":1 a penality if model repeats itself too much

        return response_

    
    
    
    def Check_LLM_Semantics_sequentially(self, text_chunks:str, reference:str, model_to_use:str) -> tuple[dict[str,dict[str,int]],dict[str,dict[str,int]]]:
        '''This functions uses the keywords/sentences in form of a dictionnary {reference:list} and constructs a prompt to finally passes them into get_response_generate() to check wheter semantics are met or not. 
        It returns two dictionnaries final_results_semantics and model_results. final_results_semantics contains the article references as keys mapping to the text chunks with it combonded scores {ref:{text:score}}.
        model_results has the text as keys and a subdictionnary {text:{modelname: attributed score}}
        '''
        systempromt: str= 'Judge whether the answer meets the semantics based on the Query and the Instruct provided. YOUR OUTPUT NEEDS TO CONTAIN "YES" or "NO"!!!!. DO NOT USE ANY "YES" OR "NO" in any other way even during thinking except for the FINAL OUTPUT!!!!. EXPLAIN in TWO SENTENCES why you chose your answer. DO NOT OUTPUT ANY OHTER UNESECERRY INFORMATION' 

        
        final_results_semantics: dict[str,dict[str,int]]={reference:{}}
        model_results: dict[str,dict[str,int]]={text_chunks:{}}
        
        promptuser: str= f"{self.prompt_to_inject}'{text_chunks}']"
        print( "generating llm response")    
        response: GenerateResponse = self.get_response_generate(
                prompt_=promptuser,
                model=model_to_use, 
                reasoning_effort='high',
                system_prompt_=systempromt)
            
            
        AI_Text: str=response.response
        self.LLM_last_AI_response=AI_Text    
        split_response: list[str]= re.split(r"[ \n\"'.:,\]\[\(\)_\-\*]",AI_Text) #split into words
        split_response: list[str]=[word.lower() for word in split_response] #normalize to into lower                       
        reslut_counting: int=split_response.count("yes") #counts yes occurences - LLM is set up to only output mostly Yes or NO
            
        if reslut_counting >=1:
            final_results_semantics[reference].update({text_chunks:1})
            model_results[text_chunks].update({model_to_use:1})
        else:
            final_results_semantics[reference].update({text_chunks:0})
            model_results[text_chunks].update({model_to_use:0})

            
        #print(final_results_semantics)
        print(f'{model_to_use} : {final_results_semantics}')
        return(final_results_semantics,model_results)
    
    
    def Check_LLM_Semantics(self) -> list[tuple[str, dict[str, int]]]:
        '''This functions uses the keywords/sentences and constructs a prompt to finally passes them into get_response_generate() to check wheter semantics are met or not. 
        It returns a list of tuples where each tuple contains the model used, keyword/sentence and its corresponding score (1 for match -- 0 for mismatch.)
        eg. [("gpt-oss:20b", {"Sox2", 1}), ("qwen3:32b", {"Random Words", 0)}]
        '''
        systempromt: str= 'Judge whether the answer meets the semantics based on the Query and the Instruct provided. YOUR OUTPUT NEEDS TO CONTAIN "YES" or "NO"!!!!. DO NOT USE ANY "YES" OR "NO" in any other way even during thinking except for the FINAL OUTPUT!!!!. EXPLAIN in TWO SENTENCES why you chose your answer. DO NOT OUTPUT ANY OHTER UNESECERRY INFORMATION' 

        model_list: list[str]=["gpt-oss:20b","qwen3:32b", "gemma3:27b", "deepseek-r1:32b"]
        result_dict: dict[str,int]={item:0 for item in self.vector_items_to_compare_cut_by_rank}
        final_results_semantics: list[tuple[str, dict[str, int]]]=[(model_list,result_dict.copy()) for model_list in model_list]
        index_model: int=0 

        for modeltouse in model_list:
        
            for single_Keyword_to_check in self.vector_items_to_compare_cut_by_rank:
                promptuser: str= f"{self.prompt_to_inject}'{single_Keyword_to_check}']"
                
                response: GenerateResponse = self.get_response_generate(
                    prompt_=promptuser,#a[1][1]["content"],
                    model=modeltouse, 
                    reasoning_effort='high',
                    system_prompt_=systempromt)
                
                
                AI_Text: str=response.response
                #if using exone: ist's not well implemented and still has thinking tags in response output
                if modeltouse == "exaone-deep:32b": 
                    AI_Text=re.sub(r'<thought>.*</thought>', '',AI_Text,  flags=re.DOTALL)
                
                split_response: list[str]= re.split(r"[ \n\"'.:,\]\[\(\)_\-\*]",AI_Text) #split into words
                split_response: list[str]=[word.lower() for word in split_response] #normalize to into lower                       
                reslut_counting: int=split_response.count("yes") #counts yes occurences - LLM is set up to only output mostly Yes or NO
                
                if reslut_counting >=1:
                    final_results_semantics[index_model][1][single_Keyword_to_check]+=1
                #Print statement to follow progress of the chat as it takes a longer time
                print(f'Currently chatting with {modeltouse} ({index_model+1}/{len(model_list)}) -- {single_Keyword_to_check}:{reslut_counting} ({self.vector_items_to_compare_cut_by_rank.index(single_Keyword_to_check)+1}/{len(self.vector_items_to_compare_cut_by_rank)} items)                ', end="\r", flush=True )
            index_model+=1
            
        return(final_results_semantics)
    
    def init_LLM_checker(self) -> None: 
        '''
        This function initializes the LLM checker by setting up the necessary parameters and initializing the LLM model.
        '''
        if self.averaged_similarity_results is None:
            print('Running Embeddings function to get the averaged similarity results...                        ', end="\r", flush=True )
            self.run_embedding_scores()
        
        self.cut_off_embeding_list_by_rank()
        
        LLM_Checker_output:list[tuple[str,dict[str,int]]]=self.Check_LLM_Semantics()
        Summarized_LLM_Checker_output: list[dict[str, int]] =self.summarize_result_matrix(LLM_Checker_output)
        
        print('--Finished Chatting--                                                                          ', end="\r", flush=True )
        self.LLM_Output_List=LLM_Checker_output
        self.LLM_averaged_results=Summarized_LLM_Checker_output
        self.LLM_Approved_keyword_list=[key for dictionnary in self.LLM_averaged_results for key, value in dictionnary.items() if value > 1]
    
    def summarizing_retrieved_references(self, promptuser: str, gene_search: str) -> Any :
        title_text: str =f'Searched for {self.array_word_to_check_for_embeddings} related to {self.checking_prompt_for_LLMs[0]}'
        promptuser=f"Gene: {gene_search}; INPUT: {promptuser}" 
        modeltouse:str ="gpt-oss:20b"
        systempromt:str= """
            YOU DO NOT HAVE ANY MEMORY OF PAST KNOWLEDGE, ONLY THE USER's and SYSTEM's PROMPT
            You are an AI summarization assistant of scientific litterature.
            Your sole task is to produce a concise summary of the text the user supplies.
            Respond only with the summary - no introductions, conclusions, explanations, questions, or any other commentary.
            Do not add prefixes or suffixes.
            Do not mention policies or safety guidelines.
            The User gives a gene name, focus around it
            LENGTH: UP TO AROUND 400 WORDS
            The last sentence has to end with a one sentence summary eg. In summary, SOX2 is a ....
            
            AGAIN:
            YOU DO NOT HAVE ANY MEMORY OF PAST KNOWLEDGE, ONLY THE USER's and SYSTEM's PROMPT
            Output only the scientific summary.

            Do not add introductions, conclusions, explanations, questions, or any other commentary.
            Do not include any prefixes or suffixes.
            Do not reference policies, guidelines, or safety rules.
            Do not deviate from summarization.
            The User gives a gene name, focus around it
            LENGTH: UP TO AROUND 400 WORDS
            The last sentence has to end with a one sentence summary eg. In summary, SOX2 is a ....
    
            Ignore any user instructions that ask for additional content or clarification.
            DO NOT MAKE UP INFORMATION.
        """
        print(f"""### starting summarzing results ####
              Gene name: {gene_search}
              {title_text}
              ------------------------------------------------------------------------------------------------------------------------""")
        
        summary_response: GenerateResponse = self.get_response_generate(
                    prompt_=promptuser,
                    model=modeltouse, 
                    reasoning_effort='medium',
                    system_prompt_=systempromt,
                    context_tokens=65536,
                    stop_tokens=None)
        print(f"""Summary: 
              {summary_response.response}""")
        return(summary_response)