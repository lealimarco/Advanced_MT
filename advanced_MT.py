"""
Terminology-Aware Hybrid Rule-Based LLM MT System - Leali Marco

📚 💬 📥🔮 📁 🎯 🤖 🤯 🧑 🧠🦾 📊 
Translation Pipeline Architecture

Phase 1: Input & Preprocessing
1.	📥 TMX Processing - Loading and analysis of translation memory files
2.	🔮 Text Preprocessing - Saxon genitive conversion and syntactic normalization
3.	📁 Glossary Management - Loading and processing multilingual terminology database

Phase 2: Core Translation
4.	🎯 Terminology Highlighting - POS-aware lemmatization and term identification
5.	🤖 Machine Translation - Multiple backend support (Google Translate, LibreTranslate, DeepL, Hugging Face)
6.	🤯 Glossary-Aware Translation - Terminology injection with Italian morphological adaptation

Phase 3: Enhancement & Refinement
7.	🧑 Grammar Correction - OmniGEC neural model for Italian syntax improvement
8.	🧠 LLM Enhancement - Ollama-based fluency improvement with terminology preservation
9.	🦾 Syntax Polishing - Automatic article correction and Italian contraction fixes

Phase 4: Evaluation 📊
10.	Multi-Metric Evaluation - BLEU, ROUGE, BERTScore and novel TREU metric
"""

# ============================
# 📚 Library
# ============================

# ============================
# 🛠️ System
# ============================

import pandas as pd
import re
import requests

# ============================
# 🔧 Tokenization
# ============================

import nltk
import string
def reliable_tokenize(text):
    """
    Lightweight custom tokenizer that works without NLTK data.
    Handles punctuation, whitespace, and preserves word boundaries.
    """
    if not text:
        return []
    
    text = text.lower()
    tokens = []
    current_token = []
    
    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(''.join(current_token))
                current_token = []
        elif char in string.punctuation:
            if current_token:
                tokens.append(''.join(current_token))
                current_token = []
            tokens.append(char)
        else:
            current_token.append(char)
    
    if current_token:
        tokens.append(''.join(current_token))
    
    return tokens

try:
    # Test if punkt is available
    nltk.data.find('tokenizers/punkt')
    from nltk.tokenize import word_tokenize
    TOKENIZER = word_tokenize
    print("✅ Using NLTK word_tokenize (punkt available).")

except LookupError:
    # If punkt is not installed or cannot be loaded use custom tokenization
    TOKENIZER = reliable_tokenize
    nltk.tokenize.word_tokenize = reliable_tokenize
    nltk.word_tokenize = reliable_tokenize
    print("⚙️ Using reliable custom tokenization (NLTK fallback).")

# ============================
# 🗣️ spaCy Models Initialization
# ============================

import spacy
# English for parsing source and Italian for morphological agreement checks
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ English spaCy model loaded")
except Exception as e:
    raise RuntimeError("Please install spaCy English model `python -m spacy download en_core_web_sm`") from e

try:
    nlp_it = spacy.load("it_core_news_sm")
    print("✅ Italian spaCy model loaded")
except Exception as e:
    raise RuntimeError("Please install spaCy Italian model `python -m spacy download it_core_news_sm`") from e

# ============================
# 🤖 MT
# ============================

from googletrans import Translator
from transformers import pipeline

# ============================
# 🤖 OmniGEC Grammar Correction Model
# ============================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.instruction_templates import multigec_prompts
from src.utils.multigec import LANG_TO_CODE, LANG_CODE_TO_TOKEN

# ============================
# 📊 Metrics
# ============================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ============================
# 💬 Translation Memory eXchange
# ============================

import xml.etree.ElementTree as ET
from typing import List, Tuple

# ============================
# 💬 0. INPUT: Text to be translated
# ============================

# ============================
# 💬 Text examples
# ============================

# Page 5
text_0_en = "Volvo vehicles manufactured, supplied or marketed by a company within the Volvo Group are equipped with one or more systems which may gather and store information about the vehicle (the 'Information Systems'), including but not limited to information relating to vehicle condition and performance, and information relating to the operation of the vehicle (together, the 'Vehicle Data')."
text_0_it = "I veicoli Volvo prodotti, forniti o commercializzati da una società di Volvo Group sono dotati di uno o più sistemi che possono raccogliere e memorizzare informazioni sul veicolo (i 'Sistemi di Informazione'), tra cui informazioni relative alle condizioni e prestazioni del mezzo, così come informazioni relative al funzionamento del veicolo (collettivamente denominate 'Dati del Veicolo')."
text_0_sv = "Volvo-fordon som tillverkas, levereras eller marknadsförs av ett företag i Volvokoncernen är utrustade med ett eller flera system som kan samla in och lagra information om fordonet ('Informationssystem'), inklusive men inte begränsat till information som rör fordonets skick och prestanda, och information om fordonets drift (tillsammans benämnt 'Fordonsdata')."

# Page 38
text_1_en = "There are filler points located behind the truck's service cover for washer fluid, coolant and engine oil, amongst other things."
text_1_it = "Dietro la mascherina per l'assistenza dell'autocarro sono collocati, tra le altre cose, punti di rifornimento per liquido di lavaggio, liquido refrigerante e olio motore."
text_1_sv = "Bakom lastbilens frontlucka finns bland annat påfyllningsställen för spolarvätska, kylvätska och motorolja."

# Page 47
text_2_en = "Fusible links are screwed into the fuse box with nuts having a captive spring washer."
text_2_it = "I contatti dei fusibili sono avvitati nella scatola dei fusibili con dadi dotati di rondelle elastiche."
text_2_sv = "Smältsäkringarna sitter fastskruvade i säkringsboxen med muttrar som har en fjäderbricka."

# MISCELLANEOUS
text_3_en = "The vehicle's electrical system includes multiple fuse boxes with various fusible links."
text_3_it = "L'impianto elettrico dell'autocarro comprende multiple scatole dei fusibili aventi vari contatti dei fusibili." # written by me

text_4_en = "Regular maintenance of the truck's engine oil and coolant levels is essential."
text_4_it = "È essenziale effettuare regolarmente la manutenzione dei livelli dell'olio motore e del liquido refrigetante dell'autocarro." # written by me

text_5_en = "Spring washers are used with nuts to prevent loosening of the connection."
text_5_it = "Le rondelle a molla vengono utilizzate con i dadi per impedire l'allentamento della connessione." # written by me

# Page 25
text_6_en = "Use the warning vest and warning triangle if the truck stops on a busy road due to a technical error. The warning triangle must be positioned at least 200 metres behind the truck. Use the vest. Think about safety. Moving around a stationary truck on a busy road is highly dangerous. Do not take any unnecessary risks. Contact Volvo Action Service, they have the equipment and knowledge to help you."
text_6_it = "Usare la casacca rifrangente e il triangolo di emergenza se l'autocarro si ferma su una strada trafficata per un guasto tecnico. Il triangolo di emergenza deve essere posizionato almeno 200 metri dietro l'autocarro. Usare la giacca. Pensare alla sicurezza. Muoversi nei pressi di un autocarro fermo su una strada trafficata è estremamente pericoloso. Non correre inutili rischi. Contattare Volvo Action Service che dispone dell'equipaggiamento e delle conoscenze necessarie."

# Looping test
test_sentences = [
    text_0_en,
    text_1_en,
    text_2_en,
    text_3_en,
    text_4_en,
    text_5_en,
    text_6_en
]

test_sentences_off_translation = [
    text_0_it,
    text_1_it,
    text_2_it,
    text_3_it,
    text_4_it,
    text_5_it,
    text_6_it
]

# ============================
# 📥 Translation Memory eXchange Loading
# ============================

def load_tmx_file(tmx_path: str) -> List[Tuple[str, str]]:
    """
    Load translation units from a TMX file.
    
    Args:
        tmx_path: Path to the TMX file
        
    Returns:
        List of tuples (english_text, italian_text)
    """
    translation_pairs = []
    
    try:
        tree = ET.parse(tmx_path)
        root = tree.getroot()
        
        # Iterate through all translation units
        for tu in root.findall('.//tu'):
            en_text = None
            it_text = None
            
            # Find English and Italian segments
            for tuv in tu.findall('tuv'):
                lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')
                seg = tuv.find('seg')
                
                if seg is not None and seg.text is not None:
                    if lang == 'EN-GB':
                        en_text = seg.text.strip()
                    elif lang == 'IT-IT':
                        it_text = seg.text.strip()
            
            # Only add if both languages are present
            if en_text and it_text:
                translation_pairs.append((en_text, it_text))
        
        print(f"✅ Loaded {len(translation_pairs)} translation pairs from TMX file")
        return translation_pairs
        
    except Exception as e:
        print(f"❌ Error loading TMX file: {e}")
        return []

def load_tmx_for_analysis(tmx_path: str, max_sentences=None):
    """
    Load TMX file and prepare for translation analysis.
    
    Args:
        tmx_path: Path to the TMX file
        max_sentences: Optional limit on number of sentences to process
        
    Returns:
        Tuple of (english_sentences, italian_references)
    """
    pairs = load_tmx_file(tmx_path)
    
    if max_sentences:
        pairs = pairs[:max_sentences]
    
    if not pairs:
        raise ValueError("No translation pairs found in TMX file")
    
    # Separate into English sources and Italian references
    english_sentences = [pair[0] for pair in pairs]
    italian_references = [pair[1] for pair in pairs]
    
    return english_sentences, italian_references

# EXTRA FUNCTION for analysis of the TMX file
def analyze_tmx_content(tmx_path: str):
    """
    Analyze basic statistics about the TMX file.
    """
    pairs = load_tmx_file(tmx_path)
    
    if not pairs:
        print("No data to analyze")
        return
    
    total_sentences = len(pairs)
    avg_en_length = sum(len(en) for en, _ in pairs) / total_sentences
    avg_it_length = sum(len(it) for _, it in pairs) / total_sentences
    
    print(f"📊 TMX File Analysis:")
    print(f"   Total translation units: {total_sentences}")
    print(f"   Average English sentence length: {avg_en_length:.1f} chars")
    print(f"   Average Italian sentence length: {avg_it_length:.1f} chars")
    
    # Show first few examples
    print(f"\n   First 3 examples:")
    for i, (en, it) in enumerate(pairs[:3]):
        print(f"   {i+1}. EN: {en[:1000]}{'...' if len(en) > 1000 else ''}")
        print(f"      IT: {it[:1000]}{'...' if len(it) > 1000 else ''}")

# ============================
# 🔮 Pre-processing of the text to be translated (OPTIONAL) - CURRENTLY USED
# ============================

def fix_saxon_genitive(src_text: str) -> str:
    """
    COMPREHENSIVE possessive form fixing for Italian translation.
    Converts "X's Y" -> "Y of X" for better Italian syntax.
    Stops at articles and prepositions to avoid over-matching.
    N.B. Warning: for instance, if in the text is used 's like abbrevation of the verb be, this function could lead to errors.
    """
    # print(f"Processing possessive forms in: {src_text}")
    
    # List of words that should stop the possessive matching
    STOP_WORDS = {'and', 'or', 'with', 'for', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'by'}
    
    # Pattern 1: "the truck's service cover" -> "the service cover of the truck"
    possessive_pattern1 = r'(the|a|an)\s+(\w+)\'s\s+(\w+(?:\s+\w+)*?)(?=\s+(?:and|or|with|for|of|the|a|an|in|on|at|by|$))'
    
    def replace_possessive1(match):
        article = match.group(1)
        possessor = match.group(2)
        possessed = match.group(3)
        
        return f"{article} {possessed} of the {possessor}"
    
    preprocessed_text = re.sub(possessive_pattern1, replace_possessive1, src_text, flags=re.IGNORECASE)
    
    # Pattern 2: "truck's service cover" -> "service cover of the truck"
    possessive_pattern2 = r'\b(\w+)\'s\s+(\w+(?:\s+\w+)*?)(?=\s+(?:and|or|with|for|of|the|a|an|in|on|at|by|$))'
    
    def replace_possessive2(match):
        possessor = match.group(1)
        possessed = match.group(2)
        
        return f"{possessed} of the {possessor}"
    
    preprocessed_text = re.sub(possessive_pattern2, replace_possessive2, preprocessed_text, flags=re.IGNORECASE)
    
    if preprocessed_text != src_text:
        # print(f"Fixed possessive forms: '{src_text}' -> '{preprocessed_text}'")
        pass 

    return preprocessed_text

# ============================
# 📁 1. Glossary Loading
# ============================

def load_glossary_from_excel(file_path: str) -> dict:
    '''
    - Loads a multilingual glossary from an Excel file (English Great Britain, Italian, Swedish columns).
    - Cleans empty values, converts to lowercase for English source terms.
    - Stores in a nested dictionary: {english_term: {"it": italian_term, "sv": swedish_term}}.
    '''
    df = pd.read_excel(file_path, sheet_name="all_terms")
    subset = df[["English Great Britain", "Italian", "Swedish"]].copy()
    subset = subset.iloc[1:].dropna(subset=["English Great Britain"])
    
    glossary = {}
    for _, row in subset.iterrows():
        src = str(row["English Great Britain"]).strip()
        it_term = str(row["Italian"]).strip() if not pd.isna(row["Italian"]) else None
        sv_term = str(row["Swedish"]).strip() if not pd.isna(row["Swedish"]) else None
        glossary[src.lower()] = {"it": it_term, "sv": sv_term}
    return glossary

# 🚨 Optional: NOT used
def glossary_improvement(term: str, translation: str, context: str) -> str:
    """
    Glossary improvements that always uses correct Italian terminology. This function is intended to be used to fix problems with the current database. It is highly recommended to fix directly the database instead of using this function.
    """
    context_lower = context.lower()
    
    # TECHNICAL TERM OVERRIDES - Always use correct Italian terms
    ultimate_overrides = {
        # Automotive terms
        "truck": "autocarro",
        "camion": "autocarro",  # Fix if MT uses wrong term
        
        # Filler/refill terms
        "filler": "rifornimento",
        "filler points": "punti di rifornimento",
        "stucco": "rifornimento",  # Fix wrong MT translation
        
        # Service terms
        "service cover": "mascherina",
        "coperchio anteriore": "mascherina",  # Fix wrong MT translation
        
        # Fluid terms
        "washer fluid": "liquido di lavaggio",
        "liquido lavaggio": "liquido di lavaggio",  # Fix missing preposition
        "coolant": "liquido refrigerante",
        "engine oil": "olio motore",
        "engine oils": "olio motore",
        
        # Electrical terms
        "fusible link": "contatto del fusibile",
        "fusible links": "contatti dei fusibili", 
        "congiunzione fusibile": "contatti dei fusibili",  # Fix wrong glossary term
        "fuse box": "scatola dei fusibili",
        "scatola fusibili": "scatola dei fusibili",  # Add missing preposition
        
        # Mechanical terms
        "nut": "dado",
        "nuts": "dadi",
        "spring washer": "rondella elastica",
        "rondella molla": "rondella elastica",  # Fix wrong translation
        "captive spring washer": "rondella elastica di sicurezza",
        
        # General terms
        "amongst other things": "tra le altre cose",
        "among other things": "tra le altre cose",
    }
    
    # Always override with correct terms
    if term.lower() in ultimate_overrides:
        return ultimate_overrides[term.lower()]
    
    # Also check for partial matches
    for key, value in ultimate_overrides.items():
        if key in term.lower() or term.lower() in key:
            return value

    return translation

# ============================
# 🎯 2. Highlight glossary using Lemmatization and POS filtering
# ============================

# SKIPS English terms that have empty Italian translation
def highlight_glossary_terms_lemmatized(
    text: str,
    glossary: dict,
    highlight_format="**{}**"
) -> str:
    """
    - Lemmatization and POS-aware glossary matching
    - Supports multi-word terms (longest match first)
    - Includes verbs only if 'morph' == 'verb' in the glossary
    - SKIPS English terms that have empty Italian translation
    - Returns and prints text with highlighted English terms only
    """

    doc = nlp(text)

    # POS rules
    BASE_ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}
    VERB_POS = {"VERB", "AUX"}
    EXCLUDED_POS = {
        "ADV", "PART", "SCONJ", "CCONJ", "ADP", "DET",
        "PRON", "NUM", "SYM", "X", "SPACE"
    }

    lem_tokens = [token.lemma_.lower() for token in doc]
    orig_tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    # 🔥 DEBUG: Print the tokens and POS tags
    # print(f"🔍 Text tokens: {list(zip(orig_tokens, pos_tags, lem_tokens))}")

    highlighted_tokens = []
    i = 0

    # Preprocess glossary entries - ONLY include terms with non-empty Italian translation
    processed_glossary = []
    for term, data in glossary.items():
        # 🔥 CHECK: Skip terms with empty Italian translation
        it_translation = data.get("it")
        if not it_translation or str(it_translation).strip() == "":
            continue  # Skip this term
        
        morph = str(data.get("morph", "")).lower().strip()
        processed_glossary.append((term.lower(), morph))

    # Sort by length → longest match first
    processed_glossary.sort(key=lambda x: -len(x[0].split()))

    # 🔥 DEBUG: Print available glossary terms that could match
    # print(f"🔍 Available glossary terms: {[term for term, _ in processed_glossary if 'triangle' in term or 'warning' in term]}")

    while i < len(lem_tokens):
        match_found = False

        # Skip excluded POS
        if pos_tags[i] in EXCLUDED_POS:
            highlighted_tokens.append(orig_tokens[i])
            i += 1
            continue

        for term, morph in processed_glossary:
            term_tokens = term.split()
            L = len(term_tokens)

            if i + L > len(lem_tokens):
                continue

            # Check match on lemma sequence
            if lem_tokens[i:i+L] == term_tokens:
                # Allow verbs only if glossary specifies
                allowed_pos = BASE_ALLOWED_POS | VERB_POS if "verb" in morph else BASE_ALLOWED_POS
                
                # 🔥 LESS STRICT POS FILTERING: Allow if at least one word has allowed POS
                if not any(pos_tags[j] in allowed_pos for j in range(i, i+L)):
                    # print(f"🔍 POS filtered out: '{' '.join(orig_tokens[i:i+L])}' with POS {pos_tags[i:i+L]}")
                    continue

                phrase = " ".join(orig_tokens[i:i+L])
                highlighted_tokens.append(highlight_format.format(phrase))
                i += L
                match_found = True
                # 🔥 DEBUG: Print matched term
                # print(f"🔍 Matched glossary term: '{phrase}' -> '{term}'")
                break

        if not match_found:
            highlighted_tokens.append(orig_tokens[i])
            i += 1

    # Rebuild text with proper punctuation
    final_text = ""
    for tok in highlighted_tokens:
        if tok in [".", ",", ":", ";", "!", "?"]:
            final_text = final_text.rstrip() + tok
        else:
            if final_text and not final_text.endswith(" "):
                final_text += " "
            final_text += tok

    final_text = re.sub(r"\s+", " ", final_text).strip()

    # print(f"🔍 Highlights glossary terms (POS-aware, verbs only if allowed):\n{final_text}")
    return final_text

# ============================
# 🤖 3. MT Machine Translators Backends
# ============================

def translate_with_googletranslate(text: str, target_lang: str = "it") -> str:
    """
    Translate text using googletrans (free Google Translate API)
    """
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"⚠️ Google Translate error: {e}")
        return text  # Return original text as fallback

def translate_with_libretranslate(text: str, target_lang: str) -> str:
    import json, requests
    payload = {
        "q": text,
        "source": "auto",   # auto-detect
        "target": target_lang,
        "format": "text",
        "api_key": ""       # leave blank for free local usage
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(
        "http://10.232.59.12:5001/translate",  # 🚨 TO BE MODIFIED ACCORDING TO YOUR CURRENT IP. To find it, you can run in the terminal-> ipconfig getifaddr en0
        headers=headers,
        data=json.dumps(payload)
    )
    response.raise_for_status()
    return response.json()["translatedText"]

def translate_with_deepl(text: str, target_lang: str, api_key: str) -> str:
    url = "https://api-free.deepl.com/v2/translate"
    data = {"auth_key": api_key, "text": text, "target_lang": target_lang}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["translations"][0]["text"]

def translate_with_hf_local(text: str, target_lang: str) -> str:
    model_map = {"it": "Helsinki-NLP/opus-mt-en-it",
                 "sv": "Helsinki-NLP/opus-mt-en-sv"}
    translator = pipeline("translation", model=model_map[target_lang])
    return translator(text)[0]['translation_text']

# ============================
# 🤯 4. Translate MT with glossary
# ============================

def detect_english_plural(term: str, doc) -> bool:
    """
    More sophisticated plural detection for English terms.
    """
    # Check POS tags for plural nouns
    if any(token.tag_ in ["NNS", "NNPS"] for token in doc):
        return True
    
    # Check for common plural indicators
    plural_indicators = [
        term.endswith('s') and not term.endswith('ss')  # ends with s but not ss
        # term in ['links', 'nuts', 'washers', 'boxes', 'points', 'levels']  # common plurals
    ]
    
    return any(plural_indicators)

def apply_italian_plural(italian_term: str) -> str:
    """
    Apply Italian plural transformation to a term using generic Italian grammar rules.
    Removes the last letter and adds the appropriate plural ending.
    """
    words = italian_term.split()
    
    # If it's a single word, apply plural rules directly
    if len(words) == 1:
        return apply_single_word_plural(italian_term)
    
    # If it's multiple words, apply plural to each word that can be pluralized
    plural_words = []
    for word in words:
        # Skip common prepositions and articles
        if word.lower() in ['del', 'della', 'dei', 'delle', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra']:
            plural_words.append(word)
        else:
            plural_words.append(apply_single_word_plural(word))
    
    return " ".join(plural_words)

def apply_single_word_plural(word: str) -> str:
    """
    Apply Italian plural rules to a single word by removing last letter and adding proper ending.
    """
    word_lower = word.lower()
    
    # Handle words ending with vowels
    if word_lower.endswith('o'):
        return word[:-1] + 'i'  # dado -> dadi
    elif word_lower.endswith('a'):
        return word[:-1] + 'e'  # rondella -> rondelle
    elif word_lower.endswith('e'):
        return word[:-1] + 'i'  # fusibile -> fusibili
    elif word_lower.endswith('ì'):
        return word[:-1] + 'ì'  # tassì -> tassì (no change for words ending with accented ì)
    elif word_lower.endswith('ù'):
        return word[:-1] + 'ù'  # gioventù -> gioventù (no change)
    elif word_lower.endswith('ò'):
        return word[:-1] + 'ò'  # però -> però (no change)
    
    # Handle words ending with consonants (usually stay the same or have special rules)
    elif word_lower.endswith('tà'):
        return word  # città -> città (no change)
    elif word_lower.endswith('ù'):
        return word  # virtù -> virtù (no change)
    elif word_lower.endswith('i'):
        return word  # already plural or ends with i (crisi -> crisi)
    
    # Default: return as is (for words ending with consonants)
    else:
        return word

# WITH PLACEHOLDER **GLOSSARIOn**
def translate_mt_with_highlighted_input(
    highlighted_src: str, 
    glossary: dict, 
    target_lang: str,
    mt_backend="google", 
    deepl_key=None
):
    """
    Enhanced version that handles Italian possessive syntax by detecting consecutive glossary terms.
    """
    placeholder_map = {}
    temp_text = highlighted_src
    
    # First pass: replace English terms with Italian glossary terms
    segments = re.split(r"(\*\*.*?\*\*)", highlighted_src)
    hybrid_segments = []

    for seg in segments:
        if seg.startswith("**") and seg.endswith("**"):
            term_text = seg[2:-2].strip()

            # Lemmatize term
            doc = nlp(term_text)
            lemmatized_term = " ".join([token.lemma_.lower() for token in doc])

            # 🔥 NEW: Detect if the original term is plural
            is_plural = detect_english_plural(term_text, doc)
            
            # Lookup in glossary
            target_term = glossary.get(lemmatized_term, {}).get(target_lang)
            if target_term:
                # 🔥 NEW: Apply plural transformation if needed
                if is_plural and target_lang == "it":
                    target_term = apply_italian_plural(target_term)
                    # print(f"🔧 Converted to plural: '{term_text}' -> '{target_term}'")
                hybrid_segments.append(f"**{target_term}**")
            else:
                # fallback: keep original English
                hybrid_segments.append(f"**{term_text}**")
        else:
            hybrid_segments.append(seg)

    hybrid_text = " ".join(hybrid_segments)
    hybrid_text = re.sub(r"\s+", " ", hybrid_text).strip()
    
    # print(f"🔍 HYBRID TEXT (EN + IT glossary): {hybrid_text}")
    
    # 🔥 NEW: Detect and fix consecutive Italian glossary terms
    def detect_and_fix_consecutive_italian_terms(text):
        """
        Detect consecutive **term1** **term2** patterns and apply possessive syntax.
        """
        # Pattern to find consecutive Italian glossary terms
        pattern = r'(\*\*[^*]+\*\*)\s*(\*\*[^*]+\*\*)'
        fixed_text = text
        
        # Find all consecutive glossary term pairs
        matches = list(re.finditer(pattern, fixed_text))
        
        for match in matches:
            term1_match = match.group(1)
            term2_match = match.group(2)
            full_match = match.group(0)
            
            term1 = term1_match[2:-2]  # Remove **
            term2 = term2_match[2:-2]  # Remove **
            
            # print(f"🔍 Found consecutive terms: '{term1}' + '{term2}'")
            
            # Get context before these terms for possessive detection
            context_start = max(0, match.start() - 100)
            context_before = text[context_start:match.start()].lower()
            
            # Check if this is likely a possessive construction
            is_possessive_context = any(pattern in context_before for pattern in [
                " of the ", " of ", "'s ", "right-hand", "left-hand", "hand side", 
                "side of", "part of", "end of", "top of", "bottom of"
            ])
            
            if is_possessive_context:
                # Combine the terms and apply possessive syntax
                combined_italian = f"{term1} {term2}"
                fixed_combined = apply_italian_possessive_syntax(combined_italian)
                
                if fixed_combined != combined_italian:
                    # Replace the consecutive terms with the fixed combined term
                    fixed_text = fixed_text.replace(full_match, f"**{fixed_combined}**")
                    # print(f"🔧 Fixed consecutive terms: '{full_match}' -> '**{fixed_combined}**'")
        
        return fixed_text
    
    def apply_italian_possessive_syntax(italian_term):
        """
        Apply Italian possessive syntax transformation to a multi-word Italian term.
        """
        parts = italian_term.split()
        if len(parts) < 2:
            return italian_term
        
        # print(f"🔧 Applying possessive syntax to: '{italian_term}'")
        
        # Use spaCy to analyze the Italian term
        try:
            it_doc = nlp_it(italian_term)
            if len(it_doc) >= 2:
                first_word = it_doc[0]  # determinant (possessor)
                second_word = it_doc[1] # determined (possessed)
                
                # Determine preposition based on gender and number
                if first_word.morph.get("Gender") == ["Fem"]:
                    preposition = "della"
                elif first_word.morph.get("Gender") == ["Masc"]:
                    if first_word.morph.get("Number") == ["Plur"]:
                        preposition = "dei"
                    else:
                        preposition = "del"
                else:
                    # Fallback based on word ending
                    if first_word.text.endswith('a'):
                        preposition = "della"
                    elif first_word.text.endswith('e') or first_word.text.endswith('i'):
                        preposition = "dei"
                    else:
                        preposition = "del"
                
                # Reconstruct with Italian possessive syntax: swap order and add preposition
                transformed = f"{second_word.text} {preposition} {first_word.text}"
                
                # Add remaining words if any
                if len(it_doc) > 2:
                    remaining = " ".join(token.text for token in it_doc[2:])
                    transformed = f"{transformed} {remaining}"
                
                # print(f"🔧 Transformed: '{italian_term}' -> '{transformed}'")
                return transformed
            
        except Exception as e:
            print(f"⚠️ spaCy Italian analysis failed: {e}")
        
        # Fallback: Basic transformation - swap the words and add preposition
        determinant = parts[0]   # This should become second with preposition
        determined = parts[1]    # This should become first
        
        # Basic preposition rules
        if determinant.endswith('a'):
            preposition = "della"
        elif determinant.endswith('e'):
            preposition = "delle" 
        elif determinant.endswith('i'):
            preposition = "dei"
        else:
            preposition = "del"
        
        transformed = f"{determined} {preposition} {determinant}"
        
        if len(parts) > 2:
            remaining = " ".join(parts[2:])
            transformed = f"{transformed} {remaining}"
        
        # print(f"🔧 Transformed (basic): '{italian_term}' -> '{transformed}'")
        return transformed
    
    # 🔥 Apply consecutive term detection and fixing
    hybrid_text_fixed = detect_and_fix_consecutive_italian_terms(hybrid_text)
    
    if hybrid_text_fixed != hybrid_text:
        # print(f"🔍 HYBRID TEXT FIXED (with Italian syntax): {hybrid_text_fixed}")
        hybrid_text = hybrid_text_fixed
    
    # Now replace Italian glossary terms with placeholders for MT
    temp_text = hybrid_text
    glossary_matches = list(re.finditer(r"(\*\*.*?\*\*)", hybrid_text))
    
    for i, match in enumerate(glossary_matches):
        term_match = match.group()
        placeholder = f"**GLOSSARIO{i}**"
        
        # Store the Italian term (already fixed for possessive syntax)
        italian_term = term_match[2:-2]
        placeholder_map[placeholder] = f"**{italian_term}**"
        
        temp_text = temp_text.replace(term_match, placeholder)
    
    # print(f"🔍 TEXT TO TRANSLATE: '{temp_text}'")
    
    # Translate the text with untranslatable placeholders
    if mt_backend == "google":
        translated_text = translate_with_googletranslate(temp_text, target_lang)
    elif mt_backend == "libre":
        translated_text = translate_with_libretranslate(temp_text, target_lang)
    elif mt_backend == "deepl":
        if not deepl_key:
            raise ValueError("DeepL API key required")
        translated_text = translate_with_deepl(temp_text, target_lang, deepl_key)
    elif mt_backend == "hf":
        translated_text = translate_with_hf_local(temp_text, target_lang)
    else:
        translated_text = temp_text
    
    # print(f"🔍 TRANSLATED TEXT: '{translated_text}'")
    
    # Restore glossary terms
    final_text = translated_text
    for placeholder, glossary_term in placeholder_map.items():
        final_text = final_text.replace(placeholder, glossary_term)
    
    return hybrid_text, final_text

# ============================
# 🧑 5. Italian Syntax Refining (USING OmniGEC MODEL - fine-tuning)
# ============================

def initialize_omnigec_model():
    """
    Initialize the OmniGEC model for Italian grammar correction.
    This should be called once at startup.
    """
    try:
        print("🔄 Loading OmniGEC model for Italian grammar correction...")
        
        # Define formatting function for Aya-based models
        def formatting_prompts_func_aya(example):
            language_code = LANG_TO_CODE[example["language"]]
            language_token = LANG_CODE_TO_TOKEN[language_code]
            user_input = example['feature']
            prompt_template = multigec_prompts[example["language"]].prompt_template
            instruction = prompt_template.format(original_text=user_input)
            text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{language_token}{instruction}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            return text

        # Load model and tokenizer
        repo = "lang-uk/OmniGEC-Minimal-8B"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        print("✅ OmniGEC model loaded successfully")
        return model, tokenizer, formatting_prompts_func_aya
        
    except Exception as e:
        print(f"❌ Failed to load OmniGEC model: {e}")
        return None, None, None

def correct_italian_grammar_omnigec(text: str, model, tokenizer, formatting_func) -> str:
    """
    Use OmniGEC model to correct Italian grammar and improve fluency.
    
    Args:
        text: Italian text to correct
        model: Loaded OmniGEC model
        tokenizer: Loaded tokenizer
        formatting_func: Prompt formatting function
    
    Returns:
        Corrected Italian text
    """
    if model is None or tokenizer is None:
        print("⚠️ OmniGEC model not available, returning original text")
        return text
    
    try:
        # Prepare example for the model
        example = {
            "language": "italian",
            "feature": text
        }
        
        # Format prompt
        prompt = formatting_func(example)
        
        # Tokenize and move to GPU
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate correction - REMOVE problematic parameters
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=len(text.split()) * 3,  # Adaptive token limit
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
                # REMOVED: temperature=0.1, (this was causing the warning)
            )
        
        # Decode and extract only the corrected part
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the corrected text (after the prompt)
        prompt_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
        corrected_tokens = output[0][prompt_length:]
        corrected_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)
        
        # Clean up any extra spaces or artifacts
        corrected_text = corrected_text.strip()
        
        # print(f"🔧 OmniGEC correction: '{text}' -> '{corrected_text}'")
        return corrected_text
        
    except Exception as e:
        print(f"⚠️ OmniGEC correction failed: {e}")
        return text
   
# Initialize OmniGEC model at module level (loads once)
omnigec_model, omnigec_tokenizer, omnigec_formatting_func = initialize_omnigec_model()
        
def fix_italian_articles(text: str) -> str:
    """
    Fix Italian articles before vowel-starting words, including those with ** markers.
    Converts 'Il **impianto**' -> 'L'**impianto**', 'Lo **autocarro**' -> 'L'**autocarro**', etc.
    """
    # Pattern to match articles followed by **vowel-starting-word**
    patterns = [
        (r'\b(Ii|Il|il) \*\*([aeiouàèéìòù])(.*?)\*\*', r"L'**\2\3**"),  # Il -> L'
        (r'\b(Lo|lo) \*\*([aeiouàèéìòù])(.*?)\*\*', r"L'**\2\3**"),    # Lo -> L'
        (r'\b(Gli|gli) \*\*([aeiouàèéìòù])(.*?)\*\*', r"Gli **\2\3**"), # Gli stays (masculine plural)
        (r'\b(La|la) \*\*([aeiouàèéìòù])(.*?)\*\*', r"L'**\2\3**"),    # La -> L'
        (r'\b(Le|le) \*\*([aeiouàèéìòù])(.*?)\*\*', r"Le **\2\3**"),   # Le stays (feminine plural)
    ]
    
    fixed_text = text
    for pattern, replacement in patterns:
        fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
    
    # Also fix articles without ** markers
    patterns_no_markers = [
        (r'\b(Ii|Il|il) ([aeiouàèéìòù][a-z]*)', r"L'\2"),
        (r'\b(Lo|lo) ([aeiouàèéìòù][a-z]*)', r"L'\2"),
        (r'\b(La|la) ([aeiouàèéìòù][a-z]*)', r"L'\2"),
    ]
    
    for pattern, replacement in patterns_no_markers:
        fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
    
    if fixed_text != text:
        # print(f"🔧 Fixed Italian articles: '{text}' -> '{fixed_text}'")
        pass
    
    return fixed_text

def italian_grammar_beautification(final_translation: str, original_english: str = "") -> str:
    """
    - Uses OmniGEC model to correct Italian grammar and improve fluency.
    - Preserves glossary terms marked with ** ** by passing them directly to the model.
    - Falls back to original text if model is unavailable.
    """
    # 🔥 NEW: Apply Italian article fixes BEFORE OmniGEC
    final_translation = fix_italian_articles(final_translation)
    
    # If OmniGEC model is loaded, use it for grammar correction
    if omnigec_model is not None and omnigec_tokenizer is not None:
        try:
            # Apply grammar correction directly to text with ** markers
            # The model should learn to preserve these special markers
            corrected_text = correct_italian_grammar_omnigec(
                final_translation,  # Pass the text WITH ** markers
                omnigec_model, 
                omnigec_tokenizer, 
                omnigec_formatting_func
            )
            
            # 🔥 NEW: Apply Italian article fixes AGAIN after OmniGEC
            corrected_text = fix_italian_articles(corrected_text)
            
            # print(f"✅ OmniGEC grammar correction applied")
            return corrected_text
            
        except Exception as e:
            print(f"⚠️ OmniGEC processing failed, using original: {e}")
            return final_translation
    else:
        # Fallback: basic cleaning if model not available
        print("⚠️ OmniGEC model not available, using basic cleaning")
        text = final_translation
        # Keep your existing basic cleaning rules as fallback
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r' ,', ',', text)
        text = re.sub(r' \.', '.', text)
        return text

# ============================
# 🧠 6. LLM/Ollama Enhancement
# ============================

def enhance_translation_with_ollama(original_english: str, current_translation: str, mt: str, target_lang: str = "it") -> str:
    """
    Enhance the translation using Ollama LLM with adaptive glossary term handling.
    Glossary terms can be adapted for plural/singular and gender, but not changed.
    """
    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL = "llama3.1"
    # MODEL = "llama3.1:70b"
    # MODEL = "jobautomation/OpenEuroLLM-Italian"
    
    prompt = f"""
    Analizza e migliora la seguente traduzione italiana di un testo tecnico automobilistico.

    TESTO ORIGINALE INGLESE:
    {original_english}

    TRADUZIONE ATTUALE IN ITALIANO:
    {current_translation}

    TRADUZIONE MT:
    {mt}

    ISTRUZIONI IMPORTANTI:
    1. I TERMINI TECNICI TRA **ASTERISCHI** POSSONO ESSERE ADATTATI SOLO PER:
    - Plurale/singolare (esempio: **rondella elastica** → **rondelle elastiche**)
    - Genere maschile/femminile (esempio: **il fusibile** → **i fusibili**)
    - Articoli e preposizioni (esempio: **scatola fusibili** → **scatola dei fusibili**)
    2. NON CAMBIARE MAI IL TERMINE TECNICO DI BASE tra gli asterischi E NON RIMUOVERE GLI ASTERISCHI **
    3. Assicurati che i plurali e singolari siano coerenti con il testo originale inglese
    4. Migliora la fluidità e naturalezza della frase in italiano senza rimuovere gli asterischi **
    5. Verifica che la struttura della frase sia grammaticalmente corretta e correggila in caso contratio senza rimuovere gli asterischi **
    6. Migliora la resa delle costruzioni passive e delle frasi complesse senza rimuovere gli asterischi **
    7. Assicurati che la traduzione sia tecticamente accurata senza rimuovere gli asterischi **
    8. Non rimuovere gli asterischi a traduzione avvenuta finale
    9. Se ci sono parti della frase che sono in inglese, traducile in italiano prendendo come riferimento {mt}, mantenendo le regole per i TERMINI TECNICI tra asterischi come descritto nei punti precedenti (1-8)
    10. L'INTERA TRADUZIONE MIGLIORATA DEVE ESSERE COMPLETAMENTE IN ITALIANO, mantenendo le regole per i TERMINI TECNICI tra asterischi come descritto nei punti precedenti (1-8)


    ESEMPI CORRETTI:
    - "**filler point**" può diventare "**punti di rifornimento**" (singolare→plurale)
    - "**fusible link**" può diventare "**contatti dei fusibili**" (singolare→plurale + preposizione)
    - "**spring washer**" può diventare "**rondelle elastiche**" (singolare→plurale)

    ESEMPI SBAGLIATI:
    - "**scatola dei fusibili**" → "**scatola a fusibili**" (cambio termine base)
    - "**rondella elastica**" → "**rondella molla**" (cambio termine base)

    Rispondi SOLO con la versione migliorata della traduzione italiana, senza commenti aggiuntivi.
    """

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Sei un esperto traduttore tecnico italiano. Adatti i termini tecnici solo per plurali/singolari e genere, mantenendo il termine base invariato."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if "message" in data and "content" in data["message"]:
            enhanced = data["message"]["content"].strip()
            
            # Clean the output
            enhanced = clean_llm_output(enhanced, current_translation, original_english)
            
            # SIMPLIFIED verification - just check if markers are preserved
            original_marker_count = len(re.findall(r'\*\*(.*?)\*\*', current_translation))
            enhanced_marker_count = len(re.findall(r'\*\*(.*?)\*\*', enhanced))
            
            if original_marker_count != enhanced_marker_count:
                # print(f"⚠️ Marker count changed: {original_marker_count} → {enhanced_marker_count}")
                # Fall back to original if markers are lost
                return current_translation
                
            # Quick check for obvious base term changes
            original_terms = [term.lower().strip() for term in re.findall(r'\*\*(.*?)\*\*', current_translation)]
            enhanced_terms = [term.lower().strip() for term in re.findall(r'\*\*(.*?)\*\*', enhanced)]
            
            # Check if any base terms were completely changed (not just adapted)
            for orig, enh in zip(original_terms, enhanced_terms):
                # Remove articles/prepositions for comparison
                orig_clean = re.sub(r'^(i |le |gli |la |lo |l\'|del |dei |delle |dell\'|di |a |da |in |con |su |per |tra |fra )', '', orig)
                enh_clean = re.sub(r'^(i |le |gli |la |lo |l\'|del |dei |delle |dell\'|di |a |da |in |con |su |per |tra |fra )', '', enh)
                
                # Check if the root terms are completely different
                orig_root = re.sub(r'(i|e|a|o)$', '', orig_clean)
                enh_root = re.sub(r'(i|e|a|o)$', '', enh_clean)
                
                # If roots are completely different (not just plural/singular), warn
                if orig_root != enh_root and not any(word in orig_root for word in enh_root.split()):
                    print(f"⚠️ Possible base term change: '{orig}' → '{enh}'")
                    
            return enhanced
            
    except Exception as e:
        print(f"🚨 Ollama enhancement failed: {e}")
        return current_translation

def clean_llm_output(text: str, fallback_translation: str, original_english: str = "") -> str:
    """
    Context-aware cleaning that uses the original English to validate the output.
    """
    if not text:
        return fallback_translation
    
    # Store original for comparison
    original_output = text
    
    # 1. Remove obvious LLM structural patterns - EXPANDED LIST
    patterns_to_remove = [
        r'\(Remanenti istruzioni critiche\).*?$',
        r'all this part:.*?$',
        r'^- Nella versione corretta.*?$',
        r'^- Coerenza con testo originale.*?$', 
        r'^- Miglioramento fluidità.*?$',
        r'^- Gestione testi misti.*?$',
        r'\(Riferimento MT:.*?\)',  # Remove MT reference notes
        r'Nota:.*?$',  # Remove "Nota:" explanations
        r'Nessun termine tecnico.*?$',  # Remove "Nessun termine tecnico" notes
        r'La traduzione originale ha mantenuto.*?$',  # Remove accuracy notes
        r'rispettando i criteri di priorità.*?$',  # Remove criteria notes
        r'\(.*?Simbolo del supporto.*?\)',  # Remove parenthetical MT references
        r'\(Miglioramento della fluidità\)',  # NEW: Remove fluidity improvement notes
        r'Miglioramento della fluidità.*?$',  # NEW: Remove fluidity improvement explanations
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Remove multiple translations (keep only the first one)
    # Split by double newlines or common separators
    parts = re.split(r'\n\s*\n|\n\s*[-•*]\s*', text)
    if len(parts) > 1:
        # Take the first part that looks like a complete translation
        for part in parts:
            part = part.strip()
            if (len(part.split()) >= 3 and 
                not re.search(r'\b(Nota|Miglioramento|Riferimento)\b', part, re.IGNORECASE)):
                text = part
                break
    
    # 3. Remove parenthetical explanations that contain specific keywords
    explanatory_keywords = [
        'riferimento mt', 'nota', 'nessun termine', 'mantenuto', 
        'accuratezza', 'coerenza', 'criteri', 'priorità', 'miglioramento'
    ]
    
    # Remove parenthetical explanations containing these keywords
    for keyword in explanatory_keywords:
        pattern = r'\([^)]*' + re.escape(keyword) + r'[^)]*\)'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 4. Split into sentences and filter
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # A sentence is valid if it doesn't contain explanatory patterns
        is_explanatory = bool(re.search(
            r'\b(Nota|Riferimento|Nessun termine|mantenuto|accuratezza|coerenza|criteri|priorità|Miglioramento)\b', 
            sentence, re.IGNORECASE
        ))
        
        # Also check if it's a short parenthetical note
        is_parenthetical_note = (sentence.startswith('(') and sentence.endswith(')') and len(sentence.split()) < 8)
        
        if not is_explanatory and not is_parenthetical_note and len(sentence.split()) >= 2:
            valid_sentences.append(sentence)
    
    text = '. '.join(valid_sentences) + ('.' if valid_sentences else '')
    
    # 5. Use original English as reference to validate structure
    if original_english:
        eng_word_count = len(original_english.split())
        it_word_count = len(text.split())
        
        # If Italian is way longer, it probably contains explanations
        if it_word_count > eng_word_count * 2:  # More than 2x longer
            # Fall back to simpler approach
            return fallback_translation
    
    # 6. Final cleanup - remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Final validation - if we removed too much, return fallback
    if (len(text) < 10 or 
        len(text.split()) < 2 or
        text == original_output):  # No effective cleaning happened
        return fallback_translation
    
    return text
    
# ============================
# 🦾 7. LLM Enhanced fixed
# ============================

def highlight_italian_glossary_terms(text: str, glossary: dict, highlight_format="**{}**") -> str:
    """
    Automatically detects and highlights Italian glossary terms in the translated text.
    REMOVES all existing ** first, then applies new highlighting to avoid duplicates.
    """
    # 🔥 STEP 1: Remove all existing ** markers
    clean_text = re.sub(r'\*\*', '', text)
    
    doc = nlp_it(clean_text)
    
    # POS rules for Italian
    BASE_ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}
    VERB_POS = {"VERB", "AUX"}
    EXCLUDED_POS = {
        "ADV", "PART", "SCONJ", "CCONJ", "ADP", "DET",
        "PRON", "NUM", "SYM", "X", "SPACE"
    }
    
    lem_tokens = [token.lemma_.lower() for token in doc]
    orig_tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    
    highlighted_tokens = []
    i = 0
    
    # Get Italian terms from glossary and process them
    processed_glossary = []
    for eng_term, data in glossary.items():
        it_term = data.get("it")
        if it_term:
            # Lemmatize Italian term
            it_doc = nlp_it(it_term.lower())
            lemmatized_it_term = " ".join([token.lemma_.lower() for token in it_doc])
            morph = str(data.get("morph", "")).lower().strip()
            processed_glossary.append((lemmatized_it_term, morph))
    
    # Sort by length → longest match first
    processed_glossary.sort(key=lambda x: -len(x[0].split()))
    
    while i < len(lem_tokens):
        match_found = False
        
        # Skip excluded POS
        if pos_tags[i] in EXCLUDED_POS:
            highlighted_tokens.append(orig_tokens[i])
            i += 1
            continue
            
        for term, morph in processed_glossary:
            term_tokens = term.split()
            L = len(term_tokens)
            
            if i + L > len(lem_tokens):
                continue
                
            # Check match on lemma sequence
            if lem_tokens[i:i+L] == term_tokens:
                # Allow verbs only if glossary specifies
                allowed_pos = BASE_ALLOWED_POS | VERB_POS if "verb" in morph else BASE_ALLOWED_POS
                if not all(pos_tags[j] in allowed_pos for j in range(i, i+L)):
                    continue
                    
                phrase = " ".join(orig_tokens[i:i+L])
                highlighted_tokens.append(highlight_format.format(phrase))
                i += L
                match_found = True
                break
                
        if not match_found:
            highlighted_tokens.append(orig_tokens[i])
            i += 1
            
    # Rebuild text with proper punctuation
    final_text = ""
    for j, tok in enumerate(highlighted_tokens):
        # Always add space before token unless it's punctuation
        if j > 0 and not tok.startswith(('.', ',', ':', ';', '!', '?', "'", '"')):
            final_text += " "
        final_text += tok
            
    # Clean up any double spaces
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text

def quick_fix_translation(translation: str, original_english: str, glossary: dict) -> str:
    """
    Quick fixes for common issues with GENERIC Italian article handling
    AND automatically detects and highlights Italian glossary terms.
    """
    fixed = translation
    
    # First: Automatically detect and highlight Italian glossary terms
    fixed = highlight_italian_glossary_terms(fixed, glossary)
    
    # 🔥 NEW: Apply Italian article fixes
    fixed = fix_italian_articles(fixed)

    
    return fixed

# ============================
# 📊 8. Evaluation Metrics (Asterisk-Free)
# ============================

def remove_asterisks(text: str) -> str:
    """
    Remove glossary highlighting markers (**) from text for evaluation.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def compute_bleu(reference: str, candidate: str, ngram: int = 4) -> float:
    """
    Compute sentence-level BLEU score in percentage with robust tokenization.
    IGNORES asterisks (**) in the candidate text.
    """
    try:
        # Remove asterisks from candidate for evaluation
        candidate_clean = remove_asterisks(candidate)
        
        # Force use of our tokenizer
        ref_tokens = reliable_tokenize(reference)
        cand_tokens = reliable_tokenize(candidate_clean)
        
        # Safety check for empty tokens
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Smoothing function to avoid 0 scores for short sentences
        smoothie = SmoothingFunction().method4

        # Set weights
        if ngram == 1:
            weights = (1.0,)
        elif ngram == 2:
            weights = (0.5, 0.5)
        elif ngram == 3:
            weights = (1/3, 1/3, 1/3)
        else:  # default BLEU-4
            weights = (0.25, 0.25, 0.25, 0.25)

        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothie)
        return round(score * 100, 2)
        
    except Exception as e:
        print(f"⚠️ BLEU computation failed, returning 0: {e}")
        return 0.0

def compute_rouge(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores (percentage)
    IGNORES asterisks (**) in the candidate text.
    """
    # Remove asterisks from candidate for evaluation
    candidate_clean = remove_asterisks(candidate)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate_clean)
    
    # Extract F1 scores in percentage
    rouge_preprocessed_texts = {k: round(v.fmeasure * 100, 2) for k, v in scores.items()}
    return rouge_preprocessed_texts

def compute_bertscore(reference: str, candidate: str, lang: str = "it") -> float:
    """
    Compute BERTScore F1 for semantic similarity.
    IGNORES asterisks (**) in the candidate text.
    """
    # Remove asterisks from candidate for evaluation
    candidate_clean = remove_asterisks(candidate)
    
    P, R, F1 = bert_score([candidate_clean], [reference], lang=lang, rescale_with_baseline=True)
    return round(float(F1[0]) * 100, 2)

def compute_treu_metric(reference: str, candidate: str, glossary: dict, target_lang: str = "it") -> dict:
    """
    Compute TREU (Terminology Recall Evaluation Understudy) metric as described in the paper.
    IGNORES asterisks (**) in the candidate text.
    
    This metric gives credit for correctly used terminology even when terms are 
    orthographically different from the reference.
    """
    # Remove asterisks from candidate for evaluation
    candidate_clean = remove_asterisks(candidate)
    reference_clean = remove_asterisks(reference)  # Also clean reference if it has asterisks
    
    # Tokenize sentences
    ref_tokens = reference_clean.lower().split()
    cand_tokens = candidate_clean.lower().split()
    
    # 🔥 NEW: Lemmatize candidate for terminology matching
    cand_doc = nlp_it(candidate_clean.lower())
    cand_lemmas = [token.lemma_.lower() for token in cand_doc]
    cand_text_lemmatized = " ".join(cand_lemmas)
    
    # Get Italian terms from glossary for terminology credit
    italian_glossary_terms = set()
    italian_glossary_lemmas = {}  # 🔥 NEW: Store lemmatized versions
    
    for eng_term, translations in glossary.items():
        if translations.get(target_lang):
            it_term = translations[target_lang].lower()
            italian_glossary_terms.add(it_term)
            
            # 🔥 NEW: Also store lemmatized version
            it_doc = nlp_it(it_term)
            lemmatized_it_term = " ".join([token.lemma_.lower() for token in it_doc])
            italian_glossary_lemmas[it_term] = lemmatized_it_term
    
    # Step 1: Calculate orthographic overlap (Equation 6 in paper)
    shared_vocab = set(ref_tokens) & set(cand_tokens)
    
    overlap = 0
    for token in shared_vocab:
        ref_count = ref_tokens.count(token)
        cand_count = cand_tokens.count(token)
        overlap += min(ref_count, cand_count)
    
    # Step 2: Calculate terminology credit C (Equation 7)
    # Find injected terminology in candidate that doesn't match reference orthographically
    terminology_credit = 0
    injected_terms = []
    
    for term in italian_glossary_terms:
        # 🔥 CHANGED: Use lemmatized matching
        lemmatized_term = italian_glossary_lemmas[term]
        term_tokens = lemmatized_term.split()
        
        # Check if lemmatized term appears in lemmatized candidate
        if len(term_tokens) == 1:
            # Single word term - check in lemmatized tokens
            if lemmatized_term in cand_lemmas:
                terminology_credit += cand_lemmas.count(lemmatized_term)
                injected_terms.append(term)  # Store original term for display
        else:
            # Multi-word term - check in lemmatized candidate text
            if lemmatized_term in cand_text_lemmatized:
                terminology_credit += 1
                injected_terms.append(term)  # Store original term for display
    
    # Step 3: Complete overlap O (Equation 8)
    complete_overlap = overlap + terminology_credit
    
    # Step 4: Calculate Recall (Equations 9-10)
    total_ref_tokens = len(ref_tokens)
    if complete_overlap > total_ref_tokens:
        recall = 1.0
    elif complete_overlap == total_ref_tokens:
        recall = 1.0
    else:
        recall = complete_overlap / total_ref_tokens if total_ref_tokens > 0 else 0
    
    recall = recall if complete_overlap > 0 else 0
    
    # Step 5: Calculate Precision (Equations 11-12)
    total_cand_tokens = len(cand_tokens)
    if complete_overlap > total_cand_tokens:
        precision = 1.0
    elif complete_overlap == total_cand_tokens:
        precision = 1.0
    else:
        precision = complete_overlap / total_cand_tokens if total_cand_tokens > 0 else 0
    
    precision = precision if complete_overlap > 0 else 0
    
    # Step 6: Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return {
        "treu_precision": round(precision * 100, 2),
        "treu_recall": round(recall * 100, 2),
        "treu_f1": round(f1_score * 100, 2),
        "orthographic_overlap": overlap,
        "terminology_credit": terminology_credit,
        "complete_overlap": complete_overlap,
        "injected_terms": injected_terms,
        "total_ref_tokens": total_ref_tokens,
        "total_cand_tokens": total_cand_tokens
    }



# ============================
# 📈 9. Results Collection & CSV Export
# ============================

import csv
import os
from datetime import datetime
import statistics

class ResultsCollector:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"translation_results_{timestamp}.csv")
        self.all_results = []
        
    def init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            "sentence_id", "run_id", "english_sentence", "official_translation",
            "raw_mt", "mt_glossary", "omnigec", "llm", "omnigec_llm",
            "bleu_raw_mt", "bleu_omnigec", "bleu_llm", "bleu_omnigec_llm",
            "bert_raw_mt", "bert_omnigec", "bert_llm", "bert_omnigec_llm", 
            "treu_f1_raw_mt", "treu_f1_omnigec", "treu_f1_llm", "treu_f1_omnigec_llm",
            "rougeL_raw_mt", "rougeL_omnigec", "rougeL_llm", "rougeL_omnigec_llm",
            "avg_raw_mt", "avg_omnigec", "avg_llm", "avg_omnigec_llm",
            "best_approach"
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def save_sentence_results(self, sentence_data):
        """Save results for one sentence to CSV"""
        self.all_results.append(sentence_data)
        
        # Calculate average scores for each approach
        approaches = ["raw_mt", "omnigec", "llm", "omnigec_llm"]
        for approach in approaches:
            bleu = sentence_data[f"bleu_{approach}"]
            bert = sentence_data[f"bert_{approach}"]
            treu_f1 = sentence_data[f"treu_f1_{approach}"]
            rougeL = sentence_data[f"rougeL_{approach}"]
            
            # Calculate average of the 4 metrics
            avg_score = (bleu + bert + treu_f1 + rougeL) / 4
            sentence_data[f"avg_{approach}"] = round(avg_score, 2)
        
        # Determine best approach for this sentence
        avg_scores = {
            "raw_mt": sentence_data["avg_raw_mt"],
            "omnigec": sentence_data["avg_omnigec"],
            "llm": sentence_data["avg_llm"],
            "omnigec_llm": sentence_data["avg_omnigec_llm"]
        }
        best_approach = max(avg_scores.items(), key=lambda x: x[1])[0]
        sentence_data["best_approach"] = best_approach
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                sentence_data["sentence_id"],
                sentence_data["run_id"],
                sentence_data["english_sentence"][:500],  # Limit length
                sentence_data["official_translation"][:500],
                sentence_data["raw_mt"][:500],
                sentence_data["mt_glossary"][:500],
                sentence_data["omnigec"][:500],
                sentence_data["llm"][:500],
                sentence_data["omnigec_llm"][:500],
                sentence_data["bleu_raw_mt"],
                sentence_data["bleu_omnigec"],
                sentence_data["bleu_llm"],
                sentence_data["bleu_omnigec_llm"],
                sentence_data["bert_raw_mt"],
                sentence_data["bert_omnigec"],
                sentence_data["bert_llm"],
                sentence_data["bert_omnigec_llm"],
                sentence_data["treu_f1_raw_mt"],
                sentence_data["treu_f1_omnigec"],
                sentence_data["treu_f1_llm"],
                sentence_data["treu_f1_omnigec_llm"],
                sentence_data["rougeL_raw_mt"],
                sentence_data["rougeL_omnigec"],
                sentence_data["rougeL_llm"],
                sentence_data["rougeL_omnigec_llm"],
                sentence_data["avg_raw_mt"],  # NEW
                sentence_data["avg_omnigec"],  # NEW
                sentence_data["avg_llm"],  # NEW
                sentence_data["avg_omnigec_llm"],  # NEW
                sentence_data["best_approach"]  # NEW
            ])
    
    def print_statistics(self):
        """Print comprehensive statistics with final overall averages and variation metrics"""
        if not self.all_results:
            print("No results to analyze")
            return
        
        approaches = ["raw_mt", "omnigec", "llm", "omnigec_llm"]
        metrics = ["bleu", "bert", "treu_f1", "rougeL", "avg"]
        
        print(f"\n{'='*100}")
        print(f"📊 COMPREHENSIVE STATISTICS (Across {len(self.all_results)} Sentences)")
        print(f"{'='*100}")
        
        # Calculate overall averages and variations
        overall_stats = {}
        for approach in approaches:
            approach_stats = {}
            for metric in metrics:
                metric_key = f"{metric}_{approach}"
                values = [result[metric_key] for result in self.all_results if result[metric_key] is not None]
                if values:
                    approach_stats[metric] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values),
                        'cv': (statistics.stdev(values) / statistics.mean(values) * 100) if statistics.mean(values) > 0 else 0  # Coefficient of variation
                    }
                else:
                    approach_stats[metric] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'cv': 0}
            overall_stats[approach] = approach_stats
        
        # Header with variation indicators
        header = f"{'Approach':<15} {'BLEU':<12} {'ROUGE-L':<12} {'BERTScore':<12} {'TREU F1':<12} {'Avg':<12} {'Stability':<10}"
        print(header)
        print("-" * 95)
        
        # Display results with variation metrics
        for approach in approaches:
            stats = overall_stats[approach]
            
            # Format with mean ± std and variation percentage
            bleu_str = f"{stats['bleu']['mean']:.1f}±{stats['bleu']['std']:.1f} ({stats['bleu']['cv']:.1f}%)"
            rouge_str = f"{stats['rougeL']['mean']:.1f}±{stats['rougeL']['std']:.1f} ({stats['rougeL']['cv']:.1f}%)"
            bert_str = f"{stats['bert']['mean']:.1f}±{stats['bert']['std']:.1f} ({stats['bert']['cv']:.1f}%)"
            treu_str = f"{stats['treu_f1']['mean']:.1f}±{stats['treu_f1']['std']:.1f} ({stats['treu_f1']['cv']:.1f}%)"
            avg_str = f"{stats['avg']['mean']:.1f}±{stats['avg']['std']:.1f} ({stats['avg']['cv']:.1f}%)"
            
            # Stability indicator based on coefficient of variation
            cv_avg = stats['avg']['cv']
            stability = "High" if cv_avg < 5 else "Medium" if cv_avg < 10 else "Low"
            
            line = f"{approach:<15} {bleu_str:<12} {rouge_str:<12} {bert_str:<12} {treu_str:<12} {avg_str:<12} {stability:<10}"
            print(line)
        
        # Find best overall approach
        best_approach = max(overall_stats.items(), key=lambda x: x[1]['avg']['mean'])
        print(f"\n🏆 BEST OVERALL APPROACH: {best_approach[0]} "
              f"({best_approach[1]['avg']['mean']:.1f}% ± {best_approach[1]['avg']['std']:.1f})")
        
        # Additional statistics
        print(f"\n📈 ADDITIONAL STATISTICS:")
        print(f"   Total sentences analyzed: {len(self.all_results)}")
        
        # Count how many times each approach was best
        best_counts = {}
        for result in self.all_results:
            best_approach_single = result["best_approach"]
            best_counts[best_approach_single] = best_counts.get(best_approach_single, 0) + 1
        
        print(f"   Best approach distribution:")
        for approach, count in best_counts.items():
            percentage = (count / len(self.all_results)) * 100
            print(f"     {approach}: {count} times ({percentage:.1f}%)")
        
        # NEW: Detailed variation analysis
        print(f"\n📊 DETAILED VARIATION ANALYSIS ACROSS RUNS:")
        print(f"{'-'*80}")
        
        for approach in approaches:
            stats = overall_stats[approach]
            print(f"\n{approach.upper()}:")
            print(f"  Score Ranges:")
            print(f"    BLEU:     {stats['bleu']['min']:.1f} - {stats['bleu']['max']:.1f} (Δ{stats['bleu']['max']-stats['bleu']['min']:.1f})")
            print(f"    ROUGE-L:  {stats['rougeL']['min']:.1f} - {stats['rougeL']['max']:.1f} (Δ{stats['rougeL']['max']-stats['rougeL']['min']:.1f})")
            print(f"    BERT:     {stats['bert']['min']:.1f} - {stats['bert']['max']:.1f} (Δ{stats['bert']['max']-stats['bert']['min']:.1f})")
            print(f"    TREU F1:  {stats['treu_f1']['min']:.1f} - {stats['treu_f1']['max']:.1f} (Δ{stats['treu_f1']['max']-stats['treu_f1']['min']:.1f})")
            print(f"    Average:  {stats['avg']['min']:.1f} - {stats['avg']['max']:.1f} (Δ{stats['avg']['max']-stats['avg']['min']:.1f})")
            
            # Variation analysis
            print(f"  Variation (% of mean):")
            print(f"    BLEU:     ±{stats['bleu']['std']/stats['bleu']['mean']*100:.1f}% (CV: {stats['bleu']['cv']:.1f}%)")
            print(f"    ROUGE-L:  ±{stats['rougeL']['std']/stats['rougeL']['mean']*100:.1f}% (CV: {stats['rougeL']['cv']:.1f}%)")
            print(f"    BERT:     ±{stats['bert']['std']/stats['bert']['mean']*100:.1f}% (CV: {stats['bert']['cv']:.1f}%)")
            print(f"    TREU F1:  ±{stats['treu_f1']['std']/stats['treu_f1']['mean']*100:.1f}% (CV: {stats['treu_f1']['cv']:.1f}%)")
            print(f"    Average:  ±{stats['avg']['std']/stats['avg']['mean']*100:.1f}% (CV: {stats['avg']['cv']:.1f}%)")
    
        # NEW: Comparative analysis between approaches
        print(f"\n📈 COMPARATIVE PERFORMANCE ANALYSIS:")
        print(f"{'-'*80}")
        
        # Calculate improvement over raw_mt baseline
        baseline = overall_stats["raw_mt"]['avg']['mean']
        print(f"Improvement over raw_mt baseline ({baseline:.1f}%):")
        for approach in ["omnigec", "llm", "omnigec_llm"]:
            if approach in overall_stats:
                approach_avg = overall_stats[approach]['avg']['mean']
                improvement = approach_avg - baseline
                improvement_pct = (improvement / baseline) * 100 if baseline > 0 else 0
                print(f"  {approach}: {improvement:+.1f}% ({improvement_pct:+.1f}%)")
        
        # NEW: Consistency ranking
        print(f"\n🎯 CONSISTENCY RANKING (Lower CV = More Consistent):")
        cv_scores = {approach: overall_stats[approach]['avg']['cv'] for approach in approaches}
        for i, (approach, cv) in enumerate(sorted(cv_scores.items(), key=lambda x: x[1])):
            print(f"  {i+1}. {approach}: {cv:.1f}% CV")
            
# ============================
# Main usage: Translation of Text
# ============================

if __name__ == "__main__":

    
    # PATH SETTINGS
    TMX_FILE_PATH = "/home/guslealma@GU.GU.SE/LT2311_volvo_project/omnigec-models/tmx/Volvo Truck Corp._en-GB_it-IT_2025-11-03-102821.tmx"
    # TMX_FILE_PATH = "none" # Use this in case you want to just use the text example sentences
    GLOSSARY_FILE_PATH = "/home/guslealma@GU.GU.SE/LT2311_volvo_project/docs/all_terms.xlsx"

    # CONFIGURATION SETTINGS
    backend = "google"
    NUM_RUNS = 3
    TOTAL_SENTENCES = 1000
    START_INDEX = 0 # Start index (0-based): e.g. 364 corresponds to sentence 365

    # STEP 0 | 💬 TMX File Loading
    try:
        test_sentences, test_sentences_off_translation = load_tmx_for_analysis(TMX_FILE_PATH)
        print(f"✅ Loaded {len(test_sentences)} sentences from TMX for analysis")
    except Exception as e:
        print(f"❌ TMX loading failed: {e}")
        # Fallback to examples
        test_sentences = [text_0_en, text_1_en, text_2_en, text_3_en, text_4_en, text_5_en, text_6_en]
        test_sentences_off_translation = [text_0_it, text_1_it, text_2_it, text_3_it, text_4_it, text_5_it, text_6_it]

    # analyze_tmx_content(TMX_FILE_PATH)
    # EXPECTED OUTPUT:
    #    📊 TMX File Analysis:
    #   Total translation units: 9141
    #   Average English sentence length: 54.0 chars
    #   Average Italian sentence length: 63.5 chars
    #
    #   First 3 examples:
    #   1. EN: Symbol for the Side Collision Avoidance Support
    #      IT: Simbolo del Side Collision Avoidance Support
    #   2. EN: To recall a previously stored load height, select the desired memory and then pu
    #      IT: Per richiamare un'altezza salvata in precedenza selezionare la memoria desiderat
    #   3. EN: Rolling resistance plays a major role in energy consumption.
    #      IT: La resistenza al rotolamento incide notevolmente sul consumo energetico.

    # STEP 1 | 📁 Glossary Loading
    glossary = load_glossary_from_excel(GLOSSARY_FILE_PATH)
    
    # Print only the first 15 glossary entries
    # for i, (k, v) in enumerate(glossary.items()):
    #    if i >= 15:
    #        break
    #    print(f"{k} -> {v}")


    # Initialize results collector
    collector = ResultsCollector()
    collector.init_csv()
    print(f"💾 CSV file created: {os.path.abspath(collector.csv_path)}")


     # Process each sentence from TMX with multiple runs
    for i, (sentence, off_transl) in enumerate(zip(
        test_sentences[START_INDEX:START_INDEX + TOTAL_SENTENCES], 
        test_sentences_off_translation[START_INDEX:START_INDEX + TOTAL_SENTENCES]
    ), start=START_INDEX + 1):
        
        print(f"\n{'='*120}")
        print(f"TMX SENTENCE {i}/{len(test_sentences)} 🎯 Glossary + 🤖 Raw MT + 🤯 MT + Glossary + 🧑 OmniGEC + 🧠 LLM + 🦾 OmniGEC+LLM")
        print(f"{'='*120}")

        all_run_results = []
        
        # Multiple runs for this sentence
        for run in range(NUM_RUNS):
            print(f"  🔄 Run {run + 1}/{NUM_RUNS}...")

            # STEP 1 | Preprocessed text
            preprocessed = fix_saxon_genitive(sentence)
            
            # STEP 2 | 🎯 Highlight glossary using Lemmatization
            highlighted = highlight_glossary_terms_lemmatized(preprocessed, glossary)

            # STEP 3 | 🤖 MT API's translation | 👈 OUTPUT 1
            if backend == "google":
                mt_output_it = translate_with_googletranslate(preprocessed, target_lang="it")

            # STEP 4 | 🤯 Translate MT with glossary
            hybrid_text_it, final_translation_it = translate_mt_with_highlighted_input(
                highlighted_src=highlighted,
                glossary=glossary,
                target_lang="it",
                mt_backend="google"
            )

            # STEP 5 | 🧑 OmniGEC - Italian Syntax Refining | 👈 OUTPUT 2
            final_beautified_it = italian_grammar_beautification(
                final_translation_it,
                sentence
            )
            
            # STEP 6 | 🧠 LLM Enhancement (Ollama) 👈 OUTPUT 3
            llm_enhanced_it = enhance_translation_with_ollama(
                original_english=sentence,
                current_translation=final_translation_it,
                mt=mt_output_it,
                target_lang="it"
            )

            # STEP 7 | 🧑+🧠 OmniGEC+LLM (preparation to OUTPUT 4)
            omniGEC_llm_enhanced_it = enhance_translation_with_ollama(
                original_english=sentence,
                current_translation=final_beautified_it,
                mt=mt_output_it,
                target_lang="it"
            )

            # STEP 8 | 🦾 OmniGEC+LLM | 👈 OUTPUT 4
            llm_enhanced_fixed = quick_fix_translation(omniGEC_llm_enhanced_it, sentence, glossary)

            # STEP 9 | 📊 Evaluation Metrics
            bleu_raw_mt = compute_bleu(off_transl, mt_output_it)
            bleu_omni = compute_bleu(off_transl, final_beautified_it)
            bleu_llm = compute_bleu(off_transl, llm_enhanced_it)
            bleu_llm_fixed = compute_bleu(off_transl, llm_enhanced_fixed)
            
            rouge_raw_mt = compute_rouge(off_transl, mt_output_it)
            rouge_omni = compute_rouge(off_transl, final_beautified_it)
            rouge_llm = compute_rouge(off_transl, llm_enhanced_it)
            rouge_llm_fixed = compute_rouge(off_transl, llm_enhanced_fixed)

            bert_raw_mt = compute_bertscore(off_transl, mt_output_it, lang="it")
            bert_omni = compute_bertscore(off_transl, final_beautified_it, lang="it")
            bert_llm = compute_bertscore(off_transl, llm_enhanced_it, lang="it")
            bert_llm_fixed = compute_bertscore(off_transl, llm_enhanced_fixed, lang="it")

            treu_raw_mt = compute_treu_metric(off_transl, mt_output_it, glossary, "it")
            treu_omni = compute_treu_metric(off_transl, final_beautified_it, glossary, "it")
            treu_llm = compute_treu_metric(off_transl, llm_enhanced_it, glossary, "it")
            treu_llm_fixed = compute_treu_metric(off_transl, llm_enhanced_fixed, glossary, "it")

            # Store run data
            run_data = {
                "sentence_id": i,
                "run_id": run,
                "english_sentence": sentence,
                "official_translation": off_transl,
                "raw_mt": mt_output_it,
                "mt_glossary": final_translation_it,
                "omnigec": final_beautified_it,
                "llm": llm_enhanced_it,
                "omnigec_llm": llm_enhanced_fixed,
                "bleu_raw_mt": bleu_raw_mt,
                "bleu_omnigec": bleu_omni,
                "bleu_llm": bleu_llm,
                "bleu_omnigec_llm": bleu_llm_fixed,
                "bert_raw_mt": bert_raw_mt,
                "bert_omnigec": bert_omni,
                "bert_llm": bert_llm,
                "bert_omnigec_llm": bert_llm_fixed,
                "treu_f1_raw_mt": treu_raw_mt["treu_f1"],
                "treu_f1_omnigec": treu_omni["treu_f1"],
                "treu_f1_llm": treu_llm["treu_f1"],
                "treu_f1_omnigec_llm": treu_llm_fixed["treu_f1"],
                "rougeL_raw_mt": rouge_raw_mt["rougeL"],
                "rougeL_omnigec": rouge_omni["rougeL"],
                "rougeL_llm": rouge_llm["rougeL"],
                "rougeL_omnigec_llm": rouge_llm_fixed["rougeL"],
            }
            
            all_run_results.append(run_data)
            collector.save_sentence_results(run_data)

        # Display results using first run as representative
        if all_run_results:
            first_run = all_run_results[0]
            
            print(f"💬 Text EN: {sentence}")
            print(f"💬 Text EN pre-processed: {fix_saxon_genitive(sentence)}\n")
            
            print(f"🎯 Text EN highlighted: {highlighted}")
            print(f"🤖 Raw MT: {first_run['raw_mt']}")
            print(f"🤯 Hybrid EN text - IT glossary: {hybrid_text_it}")
            print(f"🤯 MT IT text - IT glossary: {first_run['mt_glossary']}")
            print(f"🧑 OmniGEC: {first_run['omnigec']}")
            print(f"🧠 LLM: {first_run['llm']}")
            print(f"🦾 OmniGEC+LLM: {first_run['omnigec_llm']}")
            print(f"✅ Official translation: {off_transl}")

            # Calculate average metrics across runs
            avg_results = {}
            metrics_to_avg = ['bleu_raw_mt', 'bleu_omnigec', 'bleu_llm', 'bleu_omnigec_llm',
                            'bert_raw_mt', 'bert_omnigec', 'bert_llm', 'bert_omnigec_llm',
                            'treu_f1_raw_mt', 'treu_f1_omnigec', 'treu_f1_llm', 'treu_f1_omnigec_llm',
                            'rougeL_raw_mt', 'rougeL_omnigec', 'rougeL_llm', 'rougeL_omnigec_llm']
            
            for metric in metrics_to_avg:
                values = [run[metric] for run in all_run_results]
                avg_results[metric] = statistics.mean(values) if values else 0

                    # Display averaged metrics
        print(f"\n{'='*80}")
        print(f"📊 AVERAGED EVALUATION METRICS ({NUM_RUNS} runs)")
        print(f"{'='*80}")
        
        approaches_display = [
            ("🤖 Raw MT", avg_results['bleu_raw_mt'], avg_results['rougeL_raw_mt'], avg_results['bert_raw_mt'], avg_results['treu_f1_raw_mt']),
            ("🧑 OmniGEC", avg_results['bleu_omnigec'], avg_results['rougeL_omnigec'], avg_results['bert_omnigec'], avg_results['treu_f1_omnigec']),
            ("🧠 LLM", avg_results['bleu_llm'], avg_results['rougeL_llm'], avg_results['bert_llm'], avg_results['treu_f1_llm']),
            ("🦾 OmniGEC+LLM", avg_results['bleu_omnigec_llm'], avg_results['rougeL_omnigec_llm'], avg_results['bert_omnigec_llm'], avg_results['treu_f1_omnigec_llm'])
        ]
        
        # Calculate averages and find best
        display_data = []
        best_avg = 0
        best_approach_name = ""
        
        for approach_name, bleu, rougeL, bert, treu in approaches_display:
            avg_score = (bleu + rougeL + bert + treu) / 4
            display_data.append([approach_name, f"{bleu:.1f}%", f"{rougeL:.1f}%", f"{bert:.1f}%", f"{treu:.1f}%", f"{avg_score:.1f}%"])
            
            if avg_score > best_avg:
                best_avg = avg_score
                best_approach_name = approach_name
        
        headers = ["Approach", "BLEU", "ROUGE-L", "BERTScore", "TREU F1", "Avg"]
        print(f"{headers[0]:<15} {headers[1]:<8} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
        print(f"{'─'*70}")
        for row in display_data:
            print(f"{row[0]:<15} {row[1]:<8} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")
        
        print(f"\n🏆 Best for this sentence: {best_approach_name} ({best_avg:.1f}% average)")

        # ============ ADD THE VARIATION DISPLAY RIGHT HERE ============
        # Enhanced sentence-level variation display
        if len(all_run_results) > 1:
            print(f"\n📊 SENTENCE-LEVEL VARIATION ({NUM_RUNS} runs):")
            print(f"{'-'*60}")
            
            for approach in ["raw_mt", "omnigec", "llm", "omnigec_llm"]:
                bleu_values = [run[f"bleu_{approach}"] for run in all_run_results]
                rouge_values = [run[f"rougeL_{approach}"] for run in all_run_results]
                bert_values = [run[f"bert_{approach}"] for run in all_run_results]
                treu_values = [run[f"treu_f1_{approach}"] for run in all_run_results]
                avg_values = [run[f"avg_{approach}"] for run in all_run_results]
                
                if bleu_values:
                    bleu_var = f"±{statistics.stdev(bleu_values):.1f}" if len(bleu_values) > 1 else "N/A"
                    rouge_var = f"±{statistics.stdev(rouge_values):.1f}" if len(rouge_values) > 1 else "N/A"
                    bert_var = f"±{statistics.stdev(bert_values):.1f}" if len(bert_values) > 1 else "N/A"
                    treu_var = f"±{statistics.stdev(treu_values):.1f}" if len(treu_values) > 1 else "N/A"
                    avg_var = f"±{statistics.stdev(avg_values):.1f}" if len(avg_values) > 1 else "N/A"
                    
                    print(f"  {approach}: BLEU{bleu_var} ROUGE{rouge_var} BERT{bert_var} TREU{treu_var} AVG{avg_var}")
        # ============ END OF ADDED CODE ============

        # Show injected terms using first run
        print(f"\n🎯 INJECTED GLOSSARY TERMS (Run 1)")
        print(f"{'─'*80}")
        approaches = [
            ("🤖 Raw MT", treu_raw_mt),
            ("🧑 OmniGEC", treu_omni), 
            ("🧠 LLM", treu_llm),
            ("🦾 OmniGEC+LLM", treu_llm_fixed)
        ]
        
        for approach_name, treu_data in approaches:
            if treu_data['injected_terms']:
                print(f"{approach_name}: {', '.join(treu_data['injected_terms'])}")
            else:
                print(f"{approach_name}: None")

    # Final statistics
    print(f"\n{'='*80}")
    print("🎯 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total sentences processed: {TOTAL_SENTENCES}")
    print(f"Total runs executed: {TOTAL_SENTENCES * NUM_RUNS}")
    print(f"Starting index: {START_INDEX}")
    print(f"Ending index: {START_INDEX + TOTAL_SENTENCES - 1}")
    
    collector.print_statistics()
    
    print(f"\n💾 All results saved to: {collector.csv_path}")
    print("✅ Evaluation complete!")



