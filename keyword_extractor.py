import os
import google.generativeai as genai
from typing import List, Dict, Any, Generator, Tuple
from dotenv import load_dotenv
import re
import difflib

load_dotenv()

# --- Configure Gemini API ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

"""
You are given an arxiv taxonomy mapping which maps research topics to arxiv codes. 
    
    ARXIV_TAXONOMY = [
        "cosmology": "astro-ph.CO",
        "cosmic microwave background": "astro-ph.CO",
        "large-scale structure": "astro-ph.CO",
        "dark energy": "astro-ph.CO",
        "dark matter": "astro-ph.CO",
        "extrasolar planets": "astro-ph.EP",
        "planetary physics": "astro-ph.EP",
        "solar system": "astro-ph.EP",
        "galaxies": "astro-ph.GA",
        "milky way": "astro-ph.GA",
        "stellar populations": "astro-ph.GA",
        "active galactic nuclei": "astro-ph.GA",
        "quasars": "astro-ph.GA",
        "gamma ray bursts": "astro-ph.HE",
        "x-rays": "astro-ph.HE",
        "black holes": "astro-ph.HE",
        "neutron stars": "astro-ph.HE",
        "telescope design": "astro-ph.IM",
        "data analysis": "astro-ph.IM",
        "stellar evolution": "astro-ph.SR",
        "white dwarfs": "astro-ph.SR",
        "brown dwarfs": "astro-ph.SR",
        "solar physics": "astro-ph.SR",
        "helioseismology": "astro-ph.SR",
        "glasses": "cond-mat.dis-nn",
        "spin glasses": "cond-mat.dis-nn",
        "neural networks": "cond-mat.dis-nn",
        "nanoscale physics": "cond-mat.mes-hall",
        "graphene": "cond-mat.mes-hall",
        "quantum hall effect": "cond-mat.mes-hall",
        "spintronics": "cond-mat.mes-hall",
        "materials science": "cond-mat.mtrl-sci",
        "structural phase transitions": "cond-mat.mtrl-sci",
        "ultracold atoms": "cond-mat.quant-gas",
        "bose-einstein condensation": "cond-mat.quant-gas",
        "quantum gases": "cond-mat.quant-gas",
        "soft condensed matter": "cond-mat.soft",
        "polymers": "cond-mat.soft",
        "liquid crystals": "cond-mat.soft",
        "statistical mechanics": "cond-mat.stat-mech",
        "thermodynamics": "cond-mat.stat-mech",
        "phase transitions": "cond-mat.stat-mech",
        "strongly correlated electrons": "cond-mat.str-el",
        "quantum magnetism": "cond-mat.str-el",
        "superconductivity": "cond-mat.supr-con",
        "superflow": "cond-mat.supr-con",
        "general relativity": "gr-qc",
        "quantum cosmology": "gr-qc",
        "gravitational waves": "gr-qc",
        "high energy physics": "hep-ph",
        "phenomenology": "hep-ph",
        "theoretical particle physics": "hep-th",
        "string theory": "hep-th",
        "supersymmetry": "hep-th",
        "mathematical physics": "math-ph",
        "mathematics in physics": "math-ph",
        "adaption": "nlin.AO",
        "self-organizing systems": "nlin.AO",
        "chaotic dynamics": "nlin.CD",
        "chaos": "nlin.CD",
        "cellular automata": "nlin.CG",
        "pattern formation": "nlin.PS",
        "solitons": "nlin.PS",
        "integrable systems": "nlin.SI",
        "nuclear physics": "nucl-ex",
        "nuclear experiment": "nucl-ex",
        "nuclear theory": "nucl-th",
        "accelerator physics": "physics.acc-ph",
        "beam physics": "physics.acc-ph",
        "atmospheric physics": "physics.ao-ph",
        "oceanic physics": "physics.ao-ph",
        "climate science": "physics.ao-ph",
        "applied physics": "physics.app-ph",
        "nanotechnology": "physics.app-ph",
        "atomic clusters": "physics.atm-clus",
        "nanoparticles": "physics.atm-clus",
        "atomic physics": "physics.atom-ph",
        "cold atoms": "physics.atom-ph",
        "biological physics": "physics.bio-ph",
        "molecular biophysics": "physics.bio-ph",
        "chemical physics": "physics.chem-ph",
        "classical physics": "physics.class-ph",
        "computational physics": "physics.comp-ph",
        "data analysis": "physics.data-an",
        "fluid dynamics": "physics.flu-dyn",
        "turbulence": "physics.flu-dyn",
        "geophysics": "physics.geo-ph",
        "history of physics": "physics.hist-ph",
        "instrumentation": "physics.ins-det",
        "detectors": "physics.ins-det",
        "medical physics": "physics.med-ph",
        "biomedical imaging": "physics.med-ph",
        "optics": "physics.optics",
        "fiber optics": "physics.optics",
        "quantum optics": "physics.optics",
        "plasma physics": "physics.plasm-ph",
        "magnetically confined plasmas": "physics.plasm-ph",
        "physics and society": "physics.soc-ph",
        "space physics": "physics.space-ph",
        "quantum physics": "quant-ph",
        "quantum mechanics": "quant-ph",
        "biomolecules": "q-bio.BM",
        "dna": "q-bio.BM",
        "rna": "q-bio.BM",
        "cell behavior": "q-bio.CB",
        "genomics": "q-bio.GN",
        "dna sequencing": "q-bio.GN",
        "molecular networks": "q-bio.MN",
        "gene regulation": "q-bio.MN",
        "neurons and cognition": "q-bio.NC",
        "neural network": "q-bio.NC",
        "populations and evolution": "q-bio.PE",
        "epidemiology": "q-bio.PE",
        "quantitative methods": "q-bio.QM",
        "subcellular processes": "q-bio.SC",
        "tissues and organs": "q-bio.TO",
        "computational finance": "q-fin.CP",
        "monte carlo": "q-fin.CP",
        "economics": "q-fin.EC",
        "general finance": "q-fin.GN",
        "mathematical finance": "q-fin.MF",
        "portfolio management": "q-fin.PM",
        "pricing of securities": "q-fin.PR",
        "risk management": "q-fin.RM",
        "statistical finance": "q-fin.ST",
        "trading": "q-fin.TR",
        "market microstructure": "q-fin.TR",
        "statistics applications": "stat.AP",
        "epidemiology": "stat.AP",
        "social sciences": "stat.AP",
        "computation": "stat.CO",
        "simulation": "stat.CO",
        "methodology": "stat.ME",
        "statistical theory": "stat.TH",
        "statistical inference": "stat.TH",
        "machine learning": "cs.LG",
        "reinforcement learning": "cs.LG",
        "artificial intelligence": "cs.AI",
        "expert systems": "cs.AI",
        "hardware architecture": "cs.AR",
        "computational complexity": "cs.CC",
        "computational engineering": "cs.CE",
        "computational finance": "cs.CE",
        "computational science": "cs.CE",
        "computational geometry": "cs.CG",
        "computation and language": "cs.CL",
        "natural language processing": "cs.CL",
        "nlp": "cs.CL",
        "cryptography": "cs.CR",
        "security": "cs.CR",
        "computer vision": "cs.CV",
        "pattern recognition": "cs.CV",
        "computers and society": "cs.CY",
        "computer ethics": "cs.CY",
        "databases": "cs.DB",
        "datamining": "cs.DB",
        "distributed computing": "cs.DC",
        "parallel computing": "cs.DC",
        "cluster computing": "cs.DC",
        "digital libraries": "cs.DL",
        "discrete mathematics": "cs.DM",
        "graph theory": "cs.DM",
        "data structures": "cs.DS",
        "algorithms": "cs.DS",
        "emerging technologies": "cs.ET",
        "quantum technologies": "cs.ET",
        "formal languages": "cs.FL",
        "automata theory": "cs.FL",
        "computer graphics": "cs.GR",
        "game theory": "cs.GT",
        "human-computer interaction": "cs.HC",
        "information retrieval": "cs.IR",
        "information theory": "cs.IT",
        "logics in computer science": "cs.LO",
        "multiagent systems": "cs.MA",
        "multimedia": "cs.MM",
        "mathematical software": "cs.MS",
        "numerical analysis": "cs.NA",
        "neural networks": "cs.NE",
        "evolutionary computing": "cs.NE",
        "networking": "cs.NI",
        "internet architecture": "cs.NI",
        "operating systems": "cs.OS",
        "performance": "cs.PF",
        "programming languages": "cs.PL",
        "robotics": "cs.RO",
        "symbolic computation": "cs.SC",
        "sound": "cs.SD",
        "software engineering": "cs.SE",
        "social and information networks": "cs.SI",
        "systems and control": "cs.SY",
        "econometrics": "econ.EM",
        "general economics": "econ.GN",
        "theoretical economics": "econ.TH",
        "audio and speech processing": "eess.AS",
        "image and video processing": "eess.IV",
        "signal processing": "eess.SP",
        "systems and control": "eess.SY",
        "commutative algebra": "math.AC",
        "algebraic geometry": "math.AG",
        "analysis of pdes": "math.AP",
        "algebraic topology": "math.AT",
        "classical analysis": "math.CA",
        "combinatorics": "math.CO",
        "category theory": "math.CT",
        "complex variables": "math.CV",
        "differential geometry": "math.DG",
        "dynamical systems": "math.DS",
        "functional analysis": "math.FA",
        "general mathematics": "math.GM",
        "general topology": "math.GN",
        "group theory": "math.GR",
        "geometric topology": "math.GT",
        "history and overview": "math.HO",
        "k-theory": "math.KT",
        "logic": "math.LO",
        "metric geometry": "math.MG",
        "mathematical physics": "math.MP",
        "numerical analysis": "math.NA",
        "number theory": "math.NT",
        "operator algebras": "math.OA",
        "optimization and control": "math.OC",
        "probability": "math.PR",
        "quantum algebra": "math.QA",
        "rings and algebras": "math.RA",
        "representation theory": "math.RT",
        "symplectic geometry": "math.SG",
        "spectral theory": "math.SP",
        "statistics theory": "math.ST"
    ]

    The topic need not be synthetically same, but can be semantically same. Use your advanced knowledge to map every keyword to an arxiv code.
    Return the arxiv codes as a comma-seperated list with no duplicates.
"""


# MODIFIED FUNCTION
def extract_keywords_and_taxonomy(user_prompt: str) -> Tuple[str, Dict[str, str]]:
    """
    Uses the Gemini API to extract keywords from a user prompt and maps them to
    arXiv taxonomy codes using a fuzzy string search. Returns the raw keyword
    string and a dictionary mapping each keyword to its code.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt_template = f"""
    You are an expert at identifying keywords and topics from research paper queries.
    Given a user's prompt, extract the main topics, research areas, and keywords.
    Try to be as specific as possible with the keywords.
    Provide the keywords as a comma-separated list.

    Example format:
    - User Prompt: "latest advancements in computer vision, especially on adversarial attacks on vision transformers"
    - Keywords: Computer Vision, Adversarial Attacks, Vision Transformers

    - User Prompt: "Recent advances in deep reinforcement learning for robotics"
    - Keywords: Deep Reinforcement Learning, Robotics, Reinforcement Learning

    - User Prompt: "Efficient large language models for text generation"
    - Keywords: Large Language Models, Text Generation, NLP

    - User Prompt: "{user_prompt}"
    - Keywords:
    """

    try:
        response = model.generate_content(prompt_template)
        keywords_str = response.text.strip().replace(":", "")

        # Use original case keywords for the dictionary keys for better readability
        original_keywords = [kw.strip() for kw in keywords_str.split(',')]
        
        # keyword_code_map = {}
        # for keyword in original_keywords:
        #     # Match using the lowercased version of the keyword for consistency
        #     keyword_lower = keyword.lower()
            
        #     # Use difflib.get_close_matches for fuzzy string matching
        #     close_matches = difflib.get_close_matches(keyword_lower, ARXIV_TAXONOMY.keys(), n=1, cutoff=0.6)
            
        #     if close_matches:
        #         closest_match = close_matches[0]
        #         keyword_code_map[keyword] = ARXIV_TAXONOMY[closest_match]
        #     else:
        #         keyword_code_map[keyword] = "No match found"

        # Return a tuple with the raw keyword string and the mapping dictionary
        return keywords_str#, keyword_code_map
    except Exception as e:
        return f"Extraction failed: {e}", {}


# MODIFIED MAIN BLOCK
if __name__ == "__main__":
    # You can change the input prompt here
    input_prompt = "I need papers on ORB-SLAM3"

    keywords_list = []
    codes_list = []

    # while not codes_list and :

    extracted_keywords_str = extract_keywords_and_taxonomy(input_prompt)

    #     match_codes = re.search(r"ArXiv Codes[:\s]*(.*)", extracted_keywords_str, re.IGNORECASE)

    #     if match_codes:
    #         # Extract the part of the string after "ArXiv Codes: "
    #         codes_part = match_codes.group(1).strip()
            
    #         # Split the codes by comma and strip whitespace from each code
    #         codes_list = [code.strip() for code in codes_part.split(',')]

    keywords = [keyword.strip() for keyword in extracted_keywords_str.split(',')]

    print(f"User Prompt: {input_prompt}\n")
    print(f"Extracted Keywords String: {keywords}")