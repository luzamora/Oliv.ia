import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import warnings
from transformers import pipeline
from transformers import BartTokenizer
from dotenv import load_dotenv
import re
import random
warnings.filterwarnings('ignore')

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Falta la GOOGLE_API_KEY en el archivo .env")

from utils import (
    inferred_function,
    hypothesis_categories,
    hypothesis_attributes,
    data_merge,
    final_condition,
    top_similarities_hnsw,
    attributes_mapping
)

class Olivia:
    def __init__(self, business_data, review_data, attributes, categories, model=None, bart_summarizer=None, bart_tokenizer=None, reference="Unknown", summary_dict=None):
        self.reference = reference
        self.business_data = business_data
        self.review_data = review_data
        self.model = model if model is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.bart_summarizer = bart_summarizer if bart_summarizer is not None else pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        self.bart_tokenizer = bart_tokenizer if bart_tokenizer is not None else BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.attributes = attributes
        self.categories = categories
        self.features = None
        self.feature_embeddings = None
        self.review_embeddings = None
        self.summary = None
        self.reccomendations = None
        self.summary_cache = summary_dict if summary_dict is not None else {}

    def get_random_photo(self, business_id, df_photos):
        fotos = df_photos[df_photos["business_id"] == business_id]
        if fotos.empty:
            return None
        return fotos["photo_id"].sample()

    def get_reference(self):
        return self.reference

    def set_reference(self, reference: str):
        self.reference = reference

    def get_business_data(self):
        return self.business_data

    def set_business_data(self, business_data: pd.DataFrame):
        self.business_data = business_data

    def get_review_data(self):
        return self.review_data

    def set_review_data(self, review_data: pd.DataFrame):
        self.review_data = review_data

    def get_model(self):
        return self.model

    def set_model(self, model: SentenceTransformer):
        self.model = model

    def get_bart_summarizer(self):
        return self.bart_summarizer

    def set_bart_summarizer(self, bart_summarizer: pipeline):
        self.bart_summarizer = bart_summarizer

    def get_bart_tokenizer(self):
        return self.bart_tokenizer
    
    def set_bart_tokenizer(self, bart_tokenizer: BartTokenizer):
        self.bart_tokenizer = bart_tokenizer

    def get_attributes(self):
        return self.attributes

    def set_attributes(self, attributes: pd.DataFrame):
        self.attributes = attributes

    def get_categories(self):
        return self.categories

    def set_categories(self, categories: pd.DataFrame):
        self.categories = categories

    def get_features(self):
        return self.features

    def extract_features(self, hypothesis_attributes = hypothesis_attributes, hypothesis_categories = hypothesis_categories):
        inferred_attributes = inferred_function(self.model, hypothesis_attributes, self.review_embeddings)
        inferred_categories = inferred_function(self.model, hypothesis_categories, self.review_embeddings)

        final_attributes = final_condition(data1=self.attributes, data2=inferred_attributes)
        final_categories = final_condition(data1=self.categories, data2=inferred_categories)

        features_multihot = pd.concat([final_attributes, final_categories], axis=1)

        self.attributes = final_attributes
        self.categories = final_categories
        self.features = features_multihot
    
    def load_features(self, features):
        self.features = features

    def get_features_embeddings(self):
        return self.feature_embeddings
    
    def attributes_to_text(self, attributes_mapping = attributes_mapping):
        def binary_attributes_to_text(row, mapping):
            text = []
            for key, value in row.items():
                if key in mapping and value == 1:
                    mapped = mapping[key]
                    if callable(mapped):
                        result = mapped(value)
                        if isinstance(result, str) and result.strip():
                            text.append(result.strip())
                    elif isinstance(mapped, str):
                        text.append(mapped)
            return " | ".join(text)
        
        self.business_data["attributes_text"] = self.attributes.apply(
            lambda row: binary_attributes_to_text(row, attributes_mapping), axis=1
        )

    def fit_features_embeddings(self):
        features_text = self.features.astype(bool).apply(
            lambda row: ', '.join(self.features.columns[row.values]), axis=1
        )

        features_emb = self.model.encode(
            features_text,
            show_progress_bar=True
        )

        self.feature_embeddings = features_emb

    def load_features_embeddings(self, feature_embeddings: np.ndarray):
        self.feature_embeddings = feature_embeddings

    def get_review_embeddings(self):
        return self.review_embeddings

    def fit_reviews_embeddings(self, text_column="text", merge_id="business_id"):
        df_merge = data_merge(
            business_data=self.business_data,
            review_data=self.review_data,
            merge_id=merge_id,
            text_column=text_column
        )

        all_ind_reviews = [review for sublist in df_merge[text_column] for review in sublist]

        ind_reviews_emb = self.model.encode(
            all_ind_reviews,
            show_progress_bar=True
        )

        business_pooled_emb = []
        current_emb_idx = 0

        for index, row in df_merge.iterrows():
            num_reviews_per_business = len(row[text_column])

            start_idx = current_emb_idx
            end_idx = current_emb_idx + num_reviews_per_business

            emb_current_business = ind_reviews_emb[start_idx:end_idx]
            pooled_emb = np.mean(emb_current_business, axis = 0)

            business_pooled_emb.append(pooled_emb)

            current_emb_idx = end_idx

        self.review_embeddings = business_pooled_emb

    def load_review_embeddings(self, review_embeddings: np.ndarray):
        self.review_embeddings = review_embeddings

    def get_summaries(self):
        return self.summary_cache

    def generate_summarizer_sumy(self, reviews_list, max_chars=10_000, num_sentences=5, resumen_max_sentences=10, summarizer=LsaSummarizer()):

        if not isinstance(reviews_list, list):
            return "Input is not a list"
        texto = " ".join([str(r) for r in reviews_list if isinstance(r, str) and r.strip()])
        if not texto:
            return "Error format of list"

        bloques = []
        while len(texto) > max_chars:
            corte = texto[:max_chars].rfind(".")
            corte = corte if corte != -1 else max_chars
            bloques.append(texto[:corte + 1])
            texto = texto[corte + 1:]
        if texto.strip():
            bloques.append(texto.strip())

        resumenes = []
        for bloque in bloques:
            parser = PlaintextParser.from_string(bloque, Tokenizer("english"))
            resumen_obj = summarizer(parser.document, num_sentences)
            resumen_texto = " ".join(str(sentence) for sentence in resumen_obj)
            resumenes.append(resumen_texto)

        resumen_total = " ".join(resumenes)
        frases = resumen_total.split(". ")
        resumen_final = ". ".join(frases[:resumen_max_sentences]) + (
            '.' if not resumen_total.endswith('.') else ''
        )

        return resumen_final
    
    def generate_summarizer_bart(self, reviews_list, summary_max_sentences=10, max_tokens=1024, summary_max_length=150, summary_min_length=30):

        def chunk_division(text, max_tokens=max_tokens):
            tokens = self.bart_tokenizer.encode(text, truncation=False)
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.bart_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            return chunks

        def summarize_chunk(text):
            # Truncar para evitar superar el límite de tokens
            tokens = self.bart_tokenizer.encode(text, max_length=max_tokens, truncation=True)
            safe_text = self.bart_tokenizer.decode(tokens, skip_special_tokens=True)

            try:
                summary = self.bart_summarizer(
                    safe_text,
                    max_length=summary_max_length,
                    min_length=summary_min_length,
                    truncation=True,
                    do_sample=False
                )[0]["summary_text"]
                return summary
            except Exception as e:
                return f"[Error al resumir: {e}]"

        def summarize_long_text(text):
            if not text.strip():
                return "Texto vacío"

            tokens = self.bart_tokenizer.encode(text, truncation=False)
            if len(tokens) <= max_tokens:
                return summarize_chunk(text)

            chunks = chunk_division(text, max_tokens)
            summaries = [summarize_chunk(chunk) for chunk in chunks if chunk.strip()]
            combined_summary = " ".join(summaries)

            # Resumir nuevamente si todavía es demasiado largo
            if len(self.bart_tokenizer.encode(combined_summary, truncation=False)) > max_tokens:
                return summarize_long_text(combined_summary)
            else:
                return summarize_chunk(combined_summary)

        # Limpiar y unir las reviews
        def clean_text(text):
            import re
            text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
            return text

        text = " ".join([clean_text(str(r)) for r in reviews_list if isinstance(r, str) and r.strip()])
        summary = summarize_long_text(text)

        # Limitar a N frases
        sentences = summary.split(". ")
        final_summary = ". ".join(sentences[:summary_max_sentences]) + ('.' if not summary.endswith('.') else '')

        return final_summary
    
    def generate_summarizer_gemini(self, business_id, reviews_list, business_stars):

        def filter_and_convert_to_longtext(reviews, target_rating, tolerance = 0.5,  max_reviews = 35):
            filter_text = []
            longtext = ""

            if len(reviews) <= 35:
                filter_text = reviews
            else:
                # More than 35 reviews
                primary_reviews = []
                extended_reviews = []
                fallback_reviews = []
                pattern = r'Rated (\d\.\d) stars out of 5'

                for review in reviews:
                    if not isinstance(review, str):
                        continue
                    match = re.search(pattern, review)
                    if match:
                        rating = float(match.group(1))
                        diff = abs(rating - target_rating)
                        if diff <= tolerance:
                            primary_reviews.append(review)
                        elif diff <= 1.0:
                            extended_reviews.append(review)
                        else:
                            fallback_reviews.append(review)
                    else:
                        fallback_reviews.append(review)

                # Final list
                selected = []

                selected.extend(primary_reviews[:max_reviews])

                if len(selected) < max_reviews:
                    remaining = max_reviews - len(selected)
                    selected.extend(extended_reviews[:remaining])

                if len(selected) < max_reviews:
                    remaining = max_reviews - len(selected)
                    random.shuffle(fallback_reviews)
                    selected.extend(fallback_reviews[:remaining])
                filter_text = selected
                
            longtext = " ".join(filter_text)
            return longtext
        
        if business_id in self.summary_cache:
            return self.summary_cache[business_id]

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

        prompt = PromptTemplate(
            input_variables=["reviews"],
            template="""
            You are a helpful assistant summarizing customer reviews for restaurants.
            Summarize the following reviews in a concise paragraph that covers:
            - What customers love
            - Any frequent complaints
            - A general impression of the restaurant

            Limit your summary to a maximum of 200 words.
            
            Reviews:
            {reviews}
            
            Summary:
            """
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        text = filter_and_convert_to_longtext(reviews = reviews_list, target_rating = business_stars)
        
        if not text.strip():
            self.summary_cache[business_id] = ""
            return ""

        resumen_final = chain.run(reviews=text)
        
        self.summary_cache[business_id] = resumen_final
        return resumen_final

    def get_reccomendations(self):
        return self.reccomendations

    def reccomend(
        self,
        input_query,
        target_state,
        summarizer,
        metric=top_similarities_hnsw,
        text_column="text",
        merge_id="business_id",
    ):
        query_embedding = self.model.encode([input_query])

        if target_state == "All":
            state_mask = np.ones(len(self.business_data), dtype=bool)
        else:
            state_mask = self.business_data["state"] == target_state

        filtered_restaurants = self.business_data[state_mask].reset_index(drop=True)
        filtered_review_embeddings = np.array(self.review_embeddings)[state_mask]
        filtered_feature_embeddings = np.array(self.feature_embeddings)[state_mask]

        top_n_features = metric(
            query_embedding=query_embedding,
            comparison_embeddings=filtered_feature_embeddings,
            top_n=100
        )

        top_n_reviews = metric(
            query_embedding=query_embedding,
            comparison_embeddings=filtered_review_embeddings[top_n_features],
            top_n=20
        )

        top_n = top_n_features[top_n_reviews]

        df_merge = data_merge(
            business_data=filtered_restaurants,
            review_data=self.review_data,
            text_column=text_column,
            merge_id=merge_id
        )

        df_recommendations = df_merge.iloc[top_n]
        df_recommendations = df_recommendations.sort_values("stars", ascending=False).head(5)

        resumenes = []
        for _, row in df_recommendations.iterrows():
            if summarizer == "sumy":
                resumen = self.generate_summarizer_sumy(
                    reviews_list=row[text_column],
                    max_chars=5000,
                    num_sentences=2,
                    resumen_max_sentences=2,
                    summarizer=LsaSummarizer()
                )
                resumenes.append(resumen)
            elif summarizer == "bart":
                resumen = self.generate_summarizer_bart(
                    reviews_list=row[text_column]
                )
                resumenes.append(resumen)
            elif summarizer == "gemini":
                resumen = self.generate_summarizer_gemini(
                    business_id=row["business_id"],
                    reviews_list=row[text_column],
                    business_stars = row["stars"]
                )
                resumenes.append(resumen)
                
        df_recommendations["review_summary"] = resumenes

        self.reccomendations = df_recommendations
        