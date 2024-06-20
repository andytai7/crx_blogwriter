import os
import time

import tiktoken
from langchain.chat_models import Ollama
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import MarkdownTextSplitter

class ChatOllama:
    def __init__(self, model_name="llama3-70b", temperature=0.7):
        self.ollama = Ollama(model=model_name, temperature=temperature)
        self.conversation_history = []

    def add_to_history(self, message: str):
        """Adds a message to the conversation history."""
        self.conversation_history.append(message)

    def format_prompt(self):
        """Formats the current conversation history into a single prompt."""
        return " ".join(self.conversation_history)

    def generate_response(self, user_input):
        """Generates a response from the model based on user input and current context."""
        self.add_to_history(f"User: {user_input}")
        prompt = self.format_prompt()
        try:
            response = self.ollama._generate([prompt])
            generated_text = response.generations[0][0].text
            self.add_to_history(f"Bot: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I couldn't generate a response."

class Outliner:
    def __init__(self, input_filename=None) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_folder_path = os.path.join(script_dir, 'Transcripts')
        folder_path = default_folder_path

        transcript_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        if not transcript_files:
            raise ValueError("No transcript files found in the folder.")

        if input_filename:
            if input_filename in transcript_files:
                transcript_file_path = os.path.join(folder_path, input_filename)
            else:
                raise FileNotFoundError(f"The specified file {input_filename} does not exist in the Transcripts folder.")
        else:
            transcript_file_path = os.path.join(folder_path, transcript_files[0])

        print(f"Using transcript file: {transcript_file_path}")

        with open(transcript_file_path, "r", encoding="utf-8") as f:
            self.raw_transcript = f.read()

        self.chat = ChatOllama(model_name="llama3-70b", temperature=0.7)

    def num_tokens_from_string(
        self, string: str, encoding_name="cl100k_base"
    ) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def transcript_splitter(self, chunk_size=3000, chunk_overlap=200):
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        transcript_chunks = markdown_splitter.create_documents(
            [self.raw_transcript]
        )
        return transcript_chunks

    def transcript2insights(self, transcript):
        system_template = "You are a helpful assistant that summarizes transcripts of podcasts or lectures."
        system_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_template = """From this chunk of a presentation transcript, extract a short list of key insights. \
            Skip explaining what you're doing, labeling the insights and writing conclusion paragraphs. \
            The insights have to be phrased as statements of facts with no references to the presentation or the transcript. \
            Statements have to be full sentences and in terms of words and phrases as close as possible to those used in the transcript. \
            Keep as much detail as possible. The transcript of the presentation is delimited in triple backticks.

            Desired output format:
            - [Key insight #1]
            - [Key insight #2]
            - [...]

            Transcript:
            ```{transcript}```"""
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        )

        result = self.chat(
            chat_prompt.format_prompt(transcript=transcript).to_messages()
        )

        return result.content

    def create_essay_insights(self, transcript_chunks, verbose=True):
        response = ""
        for i, text in enumerate(transcript_chunks):
            insights = self.transcript2insights(text.page_content)
            response = "\n".join([response, insights])
            if verbose:
                print(
                    f"\nInsights extracted from chunk {i+1}/{len(transcript_chunks)}:\n{insights}"
                )
        return response

    def create_blueprint(self, statements, verbose=True):
        system_template = """You are a helpful AI blogger who writes essays on technical topics."""
        system_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )

        human_template = """Organize the provided key insights into a structured JSON outline for a blog post. Follow these steps:

        1. Identify and number each key insight.
        2. Cluster similar insights into groups, with each group containing 3-5 insights related to a specific topic.
        3. Combine these topics into higher-level themes, each consisting of 3-5 related topics.
        4. Structure these themes in a logical order, with the most overarching theme at the top as the thesis statement of the blog post.
        5. Within each theme, list the topics as supporting arguments, and under each topic, list the corresponding insights as pieces of evidence.
        6. Ensure all insights are kept as full sentences with their original wording and retain technical details.
        7. Label each layer appropriately as 'Thesis Statement', 'Supporting Arguments', and 'Evidence'.

        The output should be in JSON format, focusing solely on organizing the information hierarchically without additional explanations:

            Desired output format:
            {{
            "Thesis Statement": "...",
            "Supporting Arguments": [
                {{
                "Argument": "...",
                "Evidence": [
                    "...", "...", "...", ...
                ]
                }},
                {{
                "Argument": "...",
                "Evidence": [
                    "...", "...", "...", ...
                ]
                }},
                ...
            ]
            }}

            Statements:
            ```{statements}```"""
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        )

        outline = self.chat(
            chat_prompt.format_prompt(statements=statements).to_messages()
        )

        if verbose:
            print(f"\nEssay outline: {outline.content}\n")
        return outline.content

    # @timer_decorator
    def full_transcript2outline_json(self, verbose=True):
        print("\nChunking transcript...")
        transcript_docs = self.transcript_splitter()
        t1 = time.time()
        print("\nExtracting key insights...")
        essay_insights = self.create_essay_insights(transcript_docs, verbose)
        t2 = time.time() - t1
        print("\nCreating essay outline...")
        t1 = time.time()
        blueprint = self.create_blueprint(essay_insights, verbose)
        t3 = time.time() - t1
        if verbose:
            print()
            print(f"Extracted essay insights in {t2:.2f} seconds.")
            print(f"Created essay blueprint in {t3:.2f} seconds.")
        return blueprint
