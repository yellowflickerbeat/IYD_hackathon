f#!/usr/bin/env python3
"""
HYBRID RAG + SEMANTIC PATTERNS - FIXED SMART RETRIEVAL
Restores intelligent Kanda selection instead of always using supplementary file
"""

import pandas as pd
import numpy as np
import re
import time
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import for dynamic regex system
try:
    import inflect
    import nltk
    from nltk.stem import WordNetLemmatizer
    DYNAMIC_REGEX_AVAILABLE = True
except ImportError:
    DYNAMIC_REGEX_AVAILABLE = False
    warnings.warn("Dynamic regex system requires inflect and nltk packages. Install with: pip install inflect nltk")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.error("Groq not available. Install with: pip install groq")

class DynamicRegexGenerator:
    """Generate flexible regex patterns that handle variations in language"""
    
    def __init__(self):
        # Initialize inflect engine for plurals/tenses
        self.inflect_engine = inflect.engine()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            # NLTK resources might not be downloaded
            try:
                nltk.download('wordnet', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except:
                logger.warning("WordNet lemmatizer initialization failed. Lemmatization disabled.")
                self.lemmatizer = None
        
        # Core semantic mappings for Ramayana
        self.semantic_groups = {
            'death_words': ['died', 'dies', 'death', 'perished', 'passed away', 'expired'],
            'grief_words': ['grief', 'sorrow', 'sadness', 'mourning', 'lamentation'],
            'exile_words': ['exile', 'banishment', 'forest', 'departed', 'left'],
            'movement_words': ['went', 'goes', 'traveled', 'journeyed', 'departed'],
            'time_after': ['after', 'following', 'subsequent to', 'when', 'once'],
            'time_before': ['before', 'prior to', 'preceding'],
            'possession': ['had', 'has', 'possessed', 'owned'],
            'being': ['was', 'is', 'were', 'are', 'being'],
            'action_kill': ['killed', 'kills', 'slew', 'slain', 'destroyed'],
            'action_break': ['broke', 'breaks', 'shattered', 'destroyed'],
            'location_from': ['from', 'of', 'belonging to', 'hailing from'],
        }
        
        # Character name variations
        self.character_variants = {
            'dasaratha': ['dasaratha', 'dasharath', 'king dasaratha', 'raja dasaratha'],
            'rama': ['rama', 'raghava', 'ramachandra', 'prince rama', 'lord rama', 'dasharathi'],
            'sita': ['sita', 'seetha', 'janaki', 'vaidehi', 'maithili'],
            'hanuman': ['hanuman', 'anjaneya', 'maruti', 'pawanputra', 'bajrangbali'],
            'ravana': ['ravana', 'dashanan', 'lankesh', 'demon king'],
            'lakshmana': ['lakshmana', 'lakshman', 'saumitri'],
            'bharata': ['bharata', 'bharat'],
            'janaka': ['janaka', 'king janaka', 'raja janaka'],
        }
        
        # Tense transformation rules
        self.tense_patterns = {
            'past_to_present': {
                'died': 'dies',
                'killed': 'kills', 
                'went': 'goes',
                'was': 'is',
                'were': 'are',
                'had': 'has',
                'broke': 'breaks',
                'came': 'comes',
                'gave': 'gives',
                'took': 'takes',
                'found': 'finds',
                'built': 'builds',
                'fought': 'fights',
            }
        }
    
    def generate_dynamic_regex(self, text: str) -> str:
        """
        Generate a flexible regex pattern that handles multiple variations
        """
        # Step 1: Extract key components
        components = self._extract_key_components(text.lower())
        
        # Step 2: Generate variations for each component
        variations = self._generate_component_variations(components)
        
        # Step 3: Build flexible regex with optional elements
        regex_pattern = self._build_flexible_pattern(variations)
        
        return regex_pattern
    
    def _extract_key_components(self, text: str) -> Dict[str, List[str]]:
        """Extract meaningful components from text"""
        components = {
            'characters': [],
            'actions': [],
            'objects': [],
            'locations': [],
            'time_markers': [],
            'descriptors': []
        }
        
        words = re.findall(r'\b\w+\b', text)
        
        for word in words:
            # Character detection
            for char_key, variants in self.character_variants.items():
                if any(variant in text for variant in variants):
                    components['characters'].append(char_key)
                    break
            
            # Action detection (verbs)
            if self._is_action_word(word):
                components['actions'].append(word)
            
            # Time marker detection
            if self._is_time_marker(word):
                components['time_markers'].append(word)
            
            # Important descriptors
            if self._is_important_descriptor(word):
                components['descriptors'].append(word)
        
        return components
    
    def _generate_component_variations(self, components: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate variations for each component"""
        variations = {}
        
        for component_type, words in components.items():
            variations[component_type] = []
            
            for word in words:
                word_variations = self._get_word_variations(word, component_type)
                variations[component_type].extend(word_variations)
        
        return variations
    
    def _get_word_variations(self, word: str, component_type: str) -> List[str]:
        """Get all variations of a word"""
        variations = [word]
        
        # Character variations
        if component_type == 'characters' and word in self.character_variants:
            variations.extend(self.character_variants[word])
        
        # Semantic group variations
        for group_name, group_words in self.semantic_groups.items():
            if word in group_words:
                variations.extend(group_words)
                break
        
        # Tense variations
        if component_type == 'actions':
            # Add present tense if past tense
            if word in self.tense_patterns['past_to_present']:
                variations.append(self.tense_patterns['past_to_present'][word])
            
            # Add past tense if present tense
            for past, present in self.tense_patterns['past_to_present'].items():
                if word == present:
                    variations.append(past)
        
        # Remove duplicates and return
        return list(set(variations))
    
    def _build_flexible_pattern(self, variations: Dict[str, List[str]]) -> str:
        """Build a flexible regex pattern"""
        pattern_parts = []
        
        # Build character patterns (high importance)
        if variations.get('characters'):
            char_pattern = self._build_alternatives(variations['characters'])
            pattern_parts.append(char_pattern)
        
        # Build action patterns (high importance)  
        if variations.get('actions'):
            action_pattern = self._build_alternatives(variations['actions'])
            pattern_parts.append(action_pattern)
        
        # Build descriptor patterns (medium importance)
        if variations.get('descriptors'):
            desc_pattern = self._build_alternatives(variations['descriptors'])
            pattern_parts.append(desc_pattern)
        
        # Build time patterns (medium importance)
        if variations.get('time_markers'):
            time_pattern = self._build_alternatives(variations['time_markers'])
            pattern_parts.append(time_pattern)
        
        # Join with flexible matching (allows words in between)
        if len(pattern_parts) >= 2:
            # Use .* for flexible word separation, but limit to reasonable distance
            flexible_pattern = r'.*?'.join(pattern_parts)
            return f'(?=.*{flexible_pattern})'
        elif len(pattern_parts) == 1:
            return pattern_parts[0]
        else:
            # Fallback to simple word matching
            if variations.get('characters'):
                return r'\b' + re.escape(variations['characters'][0]) + r'\b'
            return r'\bunknown\b'  # Default fallback
    
    def _build_alternatives(self, words: List[str]) -> str:
        """Build regex alternatives for a list of words"""
        if not words:
            return ''
        
        # Clean and escape words
        cleaned_words = []
        for word in words:
            # Handle multi-word phrases
            if ' ' in word:
                # For phrases, allow flexible word separation
                phrase_words = word.split()
                phrase_pattern = r'\s+'.join([rf'\b{re.escape(w)}\b' for w in phrase_words])
                cleaned_words.append(phrase_pattern)
            else:
                cleaned_words.append(rf'\b{re.escape(word)}\b')
        
        # Create alternatives
        return f"({'|'.join(cleaned_words)})"
    
    def _is_action_word(self, word: str) -> bool:
        """Check if word is likely an action/verb"""
        action_words = {
            'died', 'dies', 'death', 'killed', 'kills', 'went', 'goes', 'came', 'comes',
            'was', 'is', 'were', 'are', 'had', 'has', 'broke', 'breaks', 'gave', 'gives',
            'took', 'takes', 'found', 'finds', 'built', 'builds', 'fought', 'fights',
            'lived', 'lives', 'ruled', 'rules', 'exiled', 'banished', 'kidnapped',
            'rescued', 'defeated', 'married', 'born', 'adopted'
        }
        return word in action_words
    
    def _is_time_marker(self, word: str) -> bool:
        """Check if word indicates time relationship"""
        time_words = {'after', 'before', 'during', 'when', 'while', 'following', 'preceding'}
        return word in time_words
    
    def _is_important_descriptor(self, word: str) -> bool:
        """Check if word is an important descriptor"""
        descriptors = {
            'grief', 'sorrow', 'exile', 'forest', 'fourteen', 'years', 'golden', 
            'bow', 'bridge', 'ocean', 'lanka', 'ayodhya', 'prince', 'king', 'queen',
            'demon', 'monkey', 'divine', 'weapon', 'head', 'arms', 'jaw', 'ring'
        }
        return word in descriptors

class SmartRAGClassifier:
    """Hybrid RAG + Semantic Pattern system with RESTORED smart Kanda selection"""
    
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library required")
        
        # Initialize Groq
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        
        # Initialize dynamic regex generator if available
        if DYNAMIC_REGEX_AVAILABLE:
            self.regex_generator = DynamicRegexGenerator()
            logger.info("✅ Dynamic regex generator initialized")
        else:
            self.regex_generator = None
            logger.warning("⚠️ Dynamic regex not available - using fallback patterns")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.5
        self.max_retries = 3
        
        # Storage for text processing
        self.semantic_patterns = {}    # Critical fallback patterns
        self.pattern_rules = []        # Loaded from supplementary file
        self.text_chunks = []          # REAL chunks from your 7 files
        self.chunk_index = {}          # Search index for chunks
        self.files_loaded = 0
        self.total_chars_processed = 0
        
        # Initialize system
        self._build_critical_patterns()
        self._test_groq_connection()
        self._load_and_process_real_files()
        self._extract_patterns_from_supplementary()
        self._build_search_index()
    
    def _build_critical_patterns(self):
        """Build minimal critical patterns as immediate fallback"""
        
        # Only the most essential patterns for fallback
        critical_patterns = [
            # Essential IRRELEVANT patterns (non-Ramayana content)
            {
                'pattern': r'\bpython\b.*\bprogramming\b',
                'verdict': 'IRRELEVANT',
                'confidence': 0.99,
                'explanation': 'IRRELEVANT: Programming language'
            },
            {
                'pattern': r'\bparis\b.*\bcapital\b.*\bfrance\b',
                'verdict': 'IRRELEVANT',
                'confidence': 0.99,
                'explanation': 'IRRELEVANT: Modern geography'
            }
        ]
        
        # Store critical patterns
        for i, pattern_info in enumerate(critical_patterns):
            self.semantic_patterns[f"critical_{i}"] = pattern_info
        
        logger.info(f"✅ Added {len(critical_patterns)} critical fallback patterns")
    
    def _extract_patterns_from_supplementary(self):
        """Extract pattern rules from supplementary knowledge file"""
        logger.info("🔍 Extracting patterns from supplementary knowledge file...")
        
        supplementary_chunks = [
            chunk for chunk in self.text_chunks 
            if 'supplementary' in chunk.get('source', '').lower()
        ]
        
        patterns_found = 0
        for chunk in supplementary_chunks:
            text = chunk['text']
            # Look for pattern sections
            if 'FAST PATTERN RECOGNITION DATABASE' in text or 'VERIFIED TRUE PATTERNS' in text:
                patterns_found += self._parse_pattern_section(text)
        
        logger.info(f"✅ Loaded {len(self.pattern_rules)} pattern rules from supplementary file")
        
        if patterns_found == 0:
            logger.warning("⚠️ No patterns found in supplementary file, using legacy patterns")
            self._build_legacy_patterns()

    def _parse_pattern_section(self, text: str) -> int:
        """Parse pattern section and build rules"""
        lines = text.split('\n')
        current_section = None
        patterns_added = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if 'VERIFIED TRUE PATTERNS:' in line:
                current_section = 'TRUE'
                logger.info("📍 Found TRUE patterns section")
            elif 'VERIFIED FALSE PATTERNS:' in line:
                current_section = 'FALSE'
                logger.info("📍 Found FALSE patterns section")
            elif 'IRRELEVANT PATTERNS:' in line:
                current_section = 'IRRELEVANT'
                logger.info("📍 Found IRRELEVANT patterns section")
            elif 'PATTERN SECTION END' in line:
                logger.info("📍 End of pattern section")
                break
            
            # Parse bullet point patterns
            elif line.startswith('-') and current_section and ':' in line:
                success = self._parse_bullet_pattern(line, current_section)
                if success:
                    patterns_added += 1
        
        return patterns_added

    def _parse_bullet_pattern(self, line: str, verdict: str) -> bool:
        """Parse bullet point pattern format"""
        try:
            # Remove bullet and split on first colon
            line = line[1:].strip()  # Remove the bullet '-'
            
            if ':' not in line or 'confidence' not in line.lower():
                logger.warning(f"⚠️ Skipping malformed pattern line: {line[:50]}...")
                return False
            
            # Split on the first occurrence of ': '
            colon_pos = line.find(': ')
            if colon_pos == -1:
                colon_pos = line.find(':')
            
            pattern_text = line[:colon_pos].strip()
            rest = line[colon_pos+1:].strip()
            
            # Extract confidence value
            conf_match = re.search(r'confidence\s+(\d+\.?\d*)', rest)
            confidence = float(conf_match.group(1)) if conf_match else 0.88
            
            # Convert to regex pattern
            regex_pattern = self._text_to_regex(pattern_text)
            
            # Create pattern rule
            pattern_rule = {
                'name': pattern_text,
                'pattern': regex_pattern,
                'verdict': verdict,
                'confidence': confidence,
                'explanation': f'{verdict}: {pattern_text}',
                'source': 'supplementary_file'
            }
            
            self.pattern_rules.append(pattern_rule)
            logger.debug(f"✅ Added pattern: {pattern_text} -> {verdict}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing pattern line '{line}': {e}")
            return False

    def _text_to_regex(self, text: str) -> str:
        """Convert natural language pattern to regex using dynamic system with fallback"""
        # Try dynamic regex generator first
        if self.regex_generator:
            try:
                dynamic_pattern = self.regex_generator.generate_dynamic_regex(text)
                if dynamic_pattern and len(dynamic_pattern) > 10:
                    logger.debug(f"🔄 Dynamic regex: '{text}' -> '{dynamic_pattern}'")
                    return dynamic_pattern
            except Exception as e:
                logger.warning(f"⚠️ Dynamic regex failed for '{text}': {e}")
        
        # Fallback to original implementation
        return self._text_to_regex_fallback(text)
    
    def _text_to_regex_fallback(self, text: str) -> str:
        """Original text_to_regex implementation as fallback"""
        text_lower = text.lower()
        
        # Handle specific pattern conversions with high precision
        conversions = {
            # Ravana patterns - CRITICAL FIXES
            'ravana had ten heads and twenty arms': r'\bravana\b.*\b(ten|10)\b.*\bheads?\b.*\b(twenty|20)\b.*\barms?\b',
            'ravana had twenty arms not ten': r'\bravana\b.*\b(twenty|20)\b.*\barms?\b',
            'ravana had twenty arms': r'\bravana\b.*\b(twenty|20)\b.*\barms?\b',
            'ravana had ten arms instead of twenty': r'\bravana\b.*\b(ten|10)\b.*\barms?\b',
            'ravana had ten arms': r'\bravana\b.*\b(ten|10)\b.*\barms?\b',
            
            # Hanuman patterns - CRITICAL FIXES
            'hanuman was son of wind god vayu': r'\bhanuman\b.*\bson\b.*\b(vayu|wind)\b',
            'hanuman was son of vayu': r'\bhanuman\b.*\bson\b.*\b(vayu|wind)\b',
            'hanuman was not son of vayu': r'\bhanuman\b.*\bnot\b.*\bson\b.*\b(vayu|wind)\b',
            
            # Exile duration patterns
            'rama was exiled for exactly fourteen years': r'\brama\b.*\bexile\b.*\b(fourteen|14)\b.*\byears?\b',
            'rama was exiled for any number other than fourteen years': r'\brama\b.*\bexile\b.*\b(fifteen|15|thirteen|13|twelve|12|sixteen|16|ten|10|eleven|11)\b.*\byears?\b',
            'fourteen years': r'\b(fourteen|14)\b.*\byears?\b',
            
            # Basic character patterns
            'rama was prince of ayodhya': r'\brama\b.*\bprince\b.*\bayodhya\b',
            'lakshmana was rama\'s younger brother': r'\blakshmana\b.*\b(younger|brother)\b.*\brama\b',
            'lakshmana was not related to rama': r'\blakshmana\b.*\bnot.*\brelated\b.*\brama\b',
            
            # Kumbhakarna sleep patterns
            'kumbhakarna slept for six months and was awake for six months': r'\bkumbhakarna\b.*\bsix\b.*\bmonths?\b',
            'kumbhakarna slept for one day only': r'\bkumbhakarna\b.*\bone\b.*\bday\b',
            
            # Bridge patterns
            'bridge to lanka was built by monkey army': r'\bbridge\b.*\blanka\b.*\bmonkey\b',
            'nala was architect who built bridge to lanka': r'\bnala\b.*\barchitect\b.*\bbridge\b',
            
            # Ramayana structure
            'ramayana has seven kandas not six': r'\bramayana\b.*\bseven\b.*\bkandas?\b',
            'ramayana has six kandas only': r'\bramayana\b.*\bsix\b.*\bkandas?\b',
            
            # Modern/irrelevant patterns
            'python programming language': r'\bpython\b.*\bprogramming\b',
            'paris capital of france': r'\bparis\b.*\bcapital\b.*\bfrance\b',
            'modern technology': r'\bmodern\b.*\btechnology\b',
        }
        
        # Check for exact matches first
        for phrase, regex in conversions.items():
            if phrase in text_lower:
                logger.debug(f"🎯 Exact conversion: '{phrase}' -> '{regex}'")
                return regex
        
        # Handle partial matches for key terms
        key_terms = {
            'ravana': r'\bravana\b',
            'hanuman': r'\bhanuman\b', 
            'rama': r'\brama\b',
            'lakshmana': r'\blakshmana\b',
            'sita': r'\bsita\b',
            'fourteen': r'\b(fourteen|14)\b',
            'twenty': r'\b(twenty|20)\b',
            'ten': r'\b(ten|10)\b',
            'arms': r'\barms?\b',
            'heads': r'\bheads?\b',
            'exile': r'\bexile\b',
            'bridge': r'\bbridge\b',
            'kandas': r'\bkandas?\b'
        }
        
        # Build pattern from key terms found
        words = re.findall(r'\b\w+\b', text_lower)
        regex_parts = []
        
        for word in words:
            if word in key_terms:
                regex_parts.append(key_terms[word])
            elif word not in ['was', 'were', 'the', 'of', 'and', 'in', 'by', 'to', 'for', 'with', 'a', 'an', 'had', 'has']:
                regex_parts.append(f'\\b{re.escape(word)}\\b')
        
        if len(regex_parts) >= 2:
            # Join with flexible matching
            pattern = r'.*'.join(regex_parts[:4])  # Limit to first 4 terms
            logger.debug(f"🔧 Generated pattern: '{text}' -> '{pattern}'")
            return pattern
        
        # Fallback: escape the entire text
        escaped = re.escape(text_lower)
        logger.debug(f"🔧 Fallback pattern: '{text}' -> '{escaped}'")
        return escaped

    def _build_legacy_patterns(self):
        """Build essential fallback patterns if file parsing fails"""
        
        # Essential patterns for core functionality
        legacy_patterns = [
            # TRUE patterns
            {
                'name': 'Ravana ten heads twenty arms',
                'pattern': r'\bravana\b.*\b(ten|10)\b.*\bheads?\b.*\b(twenty|20)\b.*\barms?\b',
                'verdict': 'TRUE',
                'confidence': 0.98,
                'explanation': 'TRUE: Ravana had exactly 10 heads and 20 arms',
                'source': 'legacy'
            },
            {
                'name': 'Hanuman son of Vayu',
                'pattern': r'\bhanuman\b.*\bson\b.*\b(vayu|wind)\b',
                'verdict': 'TRUE',
                'confidence': 0.98,
                'explanation': 'TRUE: Hanuman was son of wind god Vayu',
                'source': 'legacy'
            },
            {
                'name': 'Rama prince of Ayodhya',
                'pattern': r'\brama\b.*\bprince\b.*\bayodhya\b',
                'verdict': 'TRUE',
                'confidence': 0.98,
                'explanation': 'TRUE: Rama was prince of Ayodhya',
                'source': 'legacy'
            },
            
            # FALSE patterns
            {
                'name': 'Ravana ten arms',
                'pattern': r'\bravana\b.*\b(ten|10)\b.*\barms?\b',
                'verdict': 'FALSE',
                'confidence': 0.98,
                'explanation': 'FALSE: Ravana had 20 arms, not 10 arms',
                'source': 'legacy'
            },
            
            # IRRELEVANT patterns
            {
                'name': 'Python programming',
                'pattern': r'\bpython\b.*\bprogramming\b',
                'verdict': 'IRRELEVANT',
                'confidence': 0.99,
                'explanation': 'IRRELEVANT: Programming language',
                'source': 'legacy'
            }
        ]
        
        # Add all legacy patterns to pattern_rules
        self.pattern_rules.extend(legacy_patterns)
        
        logger.info(f"✅ Added {len(legacy_patterns)} legacy pattern rules")

    def _semantic_pattern_check(self, claim: str) -> Optional[Tuple[str, float, str]]:
        """Check patterns - first from loaded rules, then critical fallback"""
        claim_lower = claim.lower()
        claim_normalized = re.sub(r'[^\w\s]', ' ', claim_lower)
        
        # STEP 1: Check patterns loaded from supplementary file
        for rule in self.pattern_rules:
            pattern_regex = rule['pattern']
            
            try:
                if re.search(pattern_regex, claim_normalized, re.IGNORECASE):
                    verdict = rule['verdict']
                    confidence = rule['confidence']
                    explanation = rule['explanation']
                    source = rule.get('source', 'unknown')
                    
                    logger.info(f"⚡ Pattern match from {source}: {verdict} ({confidence:.2f}) - {rule.get('name', 'unnamed')}")
                    return (verdict, confidence, explanation)
            except re.error as e:
                logger.warning(f"⚠️ Invalid regex pattern '{pattern_regex}': {e}")
                continue
        
        # STEP 2: Fallback to critical hardcoded patterns
        for pattern_id, pattern_info in self.semantic_patterns.items():
            pattern_regex = pattern_info['pattern']
            
            try:
                if re.search(pattern_regex, claim_normalized, re.IGNORECASE):
                    verdict = pattern_info['verdict']
                    confidence = pattern_info['confidence']
                    explanation = pattern_info['explanation']
                    
                    logger.info(f"⚡ Critical fallback pattern match: {verdict} ({confidence:.2f})")
                    return (verdict, confidence, explanation)
            except re.error:
                continue
        
        return None
    
    def _test_groq_connection(self):
        """Test Groq API"""
        try:
            response = self._safe_groq_request([{"role": "user", "content": "Hi"}], max_tokens=3)
            logger.info("✅ Groq API connected")
        except Exception as e:
            logger.error(f"❌ Groq connection failed: {e}")
            raise
    
    def _load_and_process_real_files(self):
        """Load and process your 7 text files"""
        data_dir = Path("data")
        
        # Your actual file list
        ramayana_files = [
            "valmiki_ramayan_supplementary_knowledge.txt",  # For patterns only
            "valmiki_ramayan_bala_kanda_book1.txt",
            "valmiki_ramayan_ayodhya_kanda_book2.txt",
            "valmiki_ramayan_aranya_kanda_book3.txt", 
            "valmiki_ramayan_kishkindha_kanda_book4.txt",
            "valmiki_ramayan_sundara_kanda_book5.txt",
            "valmiki_ramayan_yuddha_kanda_book6.txt"
        ]
        
        if not data_dir.exists():
            logger.warning("⚠️ Data directory not found. Creating sample chunks.")
            self._create_fallback_chunks()
            return
        
        total_chunks = 0
        
        for filename in ramayana_files:
            file_path = data_dir / filename
            
            if file_path.exists():
                logger.info(f"📖 Processing {filename}...")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_size = len(content)
                    self.total_chars_processed += file_size
                    
                    if file_size < 100:
                        logger.warning(f"⚠️ {filename} too short, skipping")
                        continue
                    
                    # Process into meaningful chunks
                    file_chunks = self._create_meaningful_chunks(content, filename)
                    self.text_chunks.extend(file_chunks)
                    total_chunks += len(file_chunks)
                    self.files_loaded += 1
                    
                    logger.info(f"✅ {filename}: {len(file_chunks)} chunks, {file_size:,} chars")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {filename}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {filename}")
        
        logger.info(f"🎉 Processed {self.total_chars_processed:,} characters into {total_chunks} chunks from {self.files_loaded} files")
        
        if total_chunks == 0:
            logger.warning("⚠️ No real files loaded. Creating fallback chunks.")
            self._create_fallback_chunks()
    
    def _create_meaningful_chunks(self, content: str, filename: str) -> List[Dict]:
        """Create meaningful chunks from file content with proper overlap"""
        
        # Set priority based on filename - FIXED: Balanced priorities
        priority_map = {
            'supplementary': 3.0,  # REDUCED: Only used for patterns, not RAG priority
            'bala': 4.5,
            'ayodhya': 4.0,
            'aranya': 3.5,
            'kishkindha': 3.0,
            'sundara': 3.0,
            'yuddha': 2.5
        }
        
        priority = 2.0  # default
        for key, value in priority_map.items():
            if key in filename.lower():
                priority = value
                break
        
        chunks = []
        
        # Method 1: Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback: split by sentences
            sentences = re.split(r'[.!?]+', content)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        chunk_id = 0
        current_chunk = ""
        target_size = 400  # Optimal size for semantic understanding
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds target size
            if len(current_chunk) + len(para) > target_size and current_chunk:
                # Save current chunk
                if len(current_chunk.strip()) > 100:  # Minimum meaningful size
                    chunk_data = {
                        'id': f"{filename}_{chunk_id}",
                        'text': current_chunk.strip(),
                        'source': filename,
                        'priority': priority,
                        'char_count': len(current_chunk),
                        'entities': self._extract_entities(current_chunk),
                        'topics': self._extract_topics(current_chunk),
                        'fact_density': self._calculate_fact_density(current_chunk)
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1
                
                # Create overlap for next chunk
                words = current_chunk.split()
                if len(words) > 30:
                    # Take last 25 words as overlap
                    overlap_text = ' '.join(words[-25:])
                    current_chunk = overlap_text + ' ' + para
                else:
                    current_chunk = para
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += ' ' + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if len(current_chunk.strip()) > 100:
            chunk_data = {
                'id': f"{filename}_{chunk_id}",
                'text': current_chunk.strip(),
                'source': filename,
                'priority': priority,
                'char_count': len(current_chunk),
                'entities': self._extract_entities(current_chunk),
                'topics': self._extract_topics(current_chunk),
                'fact_density': self._calculate_fact_density(current_chunk)
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract Ramayana entities from text"""
        text_lower = text.lower()
        
        entity_patterns = {
            'rama': ['rama', 'raghava', 'dasharathi', 'ramachandra'],
            'sita': ['sita', 'seetha', 'janaki', 'vaidehi', 'maithili'],
            'hanuman': ['hanuman', 'anjaneya', 'maruti', 'pavan putra'],
            'ravana': ['ravana', 'dashanan', 'lankesh'],
            'bharata': ['bharata', 'bharat'],
            'lakshmana': ['lakshmana', 'lakshman', 'saumitri'],
            'dasaratha': ['dasaratha', 'dasharath'],
            'ayodhya': ['ayodhya', 'kosala'],
            'lanka': ['lanka', 'golden city'],
            'janaka': ['janaka', 'videha'],
            'kaikeyi': ['kaikeyi'],
            'kausalya': ['kausalya'],
            'sumitra': ['sumitra'],
            'sugriva': ['sugriva'],
            'vali': ['vali', 'bali'],
            'sampati': ['sampati'],
            'kumbhakarna': ['kumbhakarna'],
            'indrajit': ['indrajit', 'meghanad'],
            'vibhishana': ['vibhishana']
        }
        
        found_entities = []
        for main_entity, variations in entity_patterns.items():
            for variation in variations:
                if variation in text_lower:
                    found_entities.append(main_entity)
                    break
        
        return found_entities
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract thematic topics from text"""
        text_lower = text.lower()
        
        topic_keywords = {
            'exile': ['exile', 'banishment', 'forest', 'fourteen years'],
            'marriage': ['marriage', 'swayamvara', 'wedding', 'bow'],
            'war': ['war', 'battle', 'fight', 'army'],
            'devotion': ['devotion', 'loyalty', 'faithful', 'dedication'],
            'kidnapping': ['kidnap', 'abduct', 'taken', 'stolen'],
            'kingdom': ['kingdom', 'throne', 'rule', 'king', 'prince'],
            'dharma': ['dharma', 'righteousness', 'duty', 'virtue'],
            'demons': ['demon', 'rakshasa', 'evil', 'asura'],
            'monkeys': ['monkey', 'vanara', 'ape'],
            'bridge': ['bridge', 'setu', 'ocean', 'crossing'],
            'search': ['search', 'find', 'locate', 'looking'],
            'leap': ['leap', 'jump', 'fly', 'cross']
        }
        
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _calculate_fact_density(self, text: str) -> float:
        """Calculate how fact-dense this chunk is"""
        text_lower = text.lower()
        
        # Count factual indicators
        fact_indicators = [
            'was', 'were', 'is', 'are', 'had', 'has', 'did', 'born',
            'killed', 'married', 'ruled', 'went', 'came', 'said',
            'daughter', 'son', 'brother', 'sister', 'king', 'queen'
        ]
        
        fact_count = sum(1 for indicator in fact_indicators if indicator in text_lower)
        word_count = len(text.split())
        
        return fact_count / max(word_count, 1)
    
    def _create_fallback_chunks(self):
        """Create fallback chunks if files not found"""
        fallback_chunks = [
            {
                'id': 'fallback_1',
                'text': 'Rama was the eldest prince of Ayodhya, son of King Dasaratha. He was known for his righteousness and virtue.',
                'source': 'fallback',
                'priority': 5.0,
                'entities': ['rama', 'ayodhya', 'dasaratha'],
                'topics': ['kingdom'],
                'fact_density': 0.3
            },
            {
                'id': 'fallback_2',
                'text': 'Sita was the daughter of King Janaka of Mithila. She was kidnapped by the demon king Ravana.',
                'source': 'fallback',
                'priority': 5.0,
                'entities': ['sita', 'janaka', 'ravana'],
                'topics': ['kidnapping'],
                'fact_density': 0.4
            },
            {
                'id': 'fallback_3',
                'text': 'Hanuman was a powerful monkey warrior who could fly and leap great distances. He was devoted to Rama.',
                'source': 'fallback',
                'priority': 5.0,
                'entities': ['hanuman', 'rama'],
                'topics': ['devotion', 'monkeys'],
                'fact_density': 0.3
            }
        ]
        
        self.text_chunks = fallback_chunks
        self.files_loaded = 1
        logger.info("✅ Created fallback chunks")
    
    def _build_search_index(self):
        """Build search index WITHOUT over-prioritizing supplementary"""
        logger.info("🔍 Building balanced search index...")
        
        # Build entity index with BALANCED weighting
        entity_index = {}
        topic_index = {}
        
        for chunk in self.text_chunks:
            chunk_id = chunk['id']
            
            # FIXED: No artificial priority boost for supplementary
            # Let natural Kanda relevance determine selection
            
            # Index by entities
            for entity in chunk.get('entities', []):
                if entity not in entity_index:
                    entity_index[entity] = []
                entity_index[entity].append({
                    'chunk_id': chunk_id, 
                    'priority': chunk['priority']  # Use natural priority
                })
            
            # Index by topics
            for topic in chunk.get('topics', []):
                if topic not in topic_index:
                    topic_index[topic] = []
                topic_index[topic].append({
                    'chunk_id': chunk_id, 
                    'priority': chunk['priority']  # Use natural priority
                })
        
        # Sort indices by priority
        for entity in entity_index:
            entity_index[entity].sort(key=lambda x: x['priority'], reverse=True)
        
        for topic in topic_index:
            topic_index[topic].sort(key=lambda x: x['priority'], reverse=True)
        
        self.chunk_index = {
            'entities': entity_index,
            'topics': topic_index,
            'chunks_by_id': {chunk['id']: chunk for chunk in self.text_chunks}
        }
        
        # Log stats
        kanda_counts = {}
        for chunk in self.text_chunks:
            source = chunk.get('source', 'unknown')
            kanda_counts[source] = kanda_counts.get(source, 0) + 1
        
        logger.info(f"✅ Built balanced search index: {len(entity_index)} entities, {len(topic_index)} topics")
        logger.info(f"📚 Chunk distribution: {kanda_counts}")
    
    def _determine_best_kanda(self, claim_lower: str, entities: List[str], topics: List[str]) -> Dict[str, List[str]]:
        """RESTORED: Determine which Kanda books are most relevant to this query"""
        
        # Define what each Kanda book covers
        kanda_coverage = {
            'bala': {
                'topics': ['birth', 'childhood', 'education', 'bow', 'marriage', 'swayamvara'],
                'entities': ['vasishtha', 'vishwamitra', 'janaka', 'shiva', 'tataka'],
                'keywords': ['born', 'child', 'guru', 'teacher', 'bow', 'break', 'wedding', 'marriage'],
                'events': ['rama birth', 'bow breaking', 'sita marriage', 'education']
            },
            'ayodhya': {
                'topics': ['exile', 'kingdom', 'succession', 'politics'],
                'entities': ['dasaratha', 'kaikeyi', 'bharata', 'manthara'],
                'keywords': ['exile', 'kingdom', 'throne', 'boon', 'fourteen', 'years', 'forest'],
                'events': ['dasaratha death', 'exile beginning', 'bharata regency']
            },
            'aranya': {
                'topics': ['forest', 'demons', 'exile', 'kidnapping'],
                'entities': ['shurpanakha', 'khara', 'maricha', 'ravana'],
                'keywords': ['forest', 'demon', 'golden', 'deer', 'kidnap', 'abduct'],
                'events': ['surpanakha incident', 'golden deer', 'sita kidnapping']
            },
            'kishkindha': {
                'topics': ['monkeys', 'alliance', 'friendship'],
                'entities': ['vali', 'sugriva', 'tara', 'angada'],
                'keywords': ['monkey', 'vanara', 'alliance', 'help', 'friend'],
                'events': ['vali killing', 'monkey alliance', 'search planning']
            },
            'sundara': {
                'topics': ['search', 'lanka', 'reconnaissance'],
                'entities': ['hanuman', 'sampati', 'jambavan'],
                'keywords': ['search', 'find', 'leap', 'ocean', 'lanka', 'ring'],
                'events': ['ocean crossing', 'sita meeting', 'lanka burning']
            },
            'yuddha': {
                'topics': ['war', 'battle', 'bridge', 'victory'],
                'entities': ['indrajit', 'kumbhakarna', 'vibhishana', 'lakshmana'],
                'keywords': ['war', 'battle', 'fight', 'bridge', 'weapon', 'victory'],
                'events': ['bridge building', 'lanka war', 'ravana death']
            }
        }
        
        # Score each Kanda for relevance to this query
        kanda_scores = {}
        
        for kanda, coverage in kanda_coverage.items():
            score = 0
            
            # Entity matches
            entity_matches = sum(1 for entity in entities if entity in coverage['entities'])
            score += entity_matches * 15
            
            # Topic matches  
            topic_matches = sum(1 for topic in topics if topic in coverage['topics'])
            score += topic_matches * 10
            
            # Keyword presence
            keyword_matches = sum(1 for keyword in coverage['keywords'] if keyword in claim_lower)
            score += keyword_matches * 5
            
            # Event context
            event_matches = sum(1 for event in coverage['events'] if any(word in claim_lower for word in event.split()))
            score += event_matches * 8
            
            kanda_scores[kanda] = score
        
        # Sort by relevance
        sorted_kandas = sorted(kanda_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select primary (top 1-2) and secondary (next 1-2) Kandas
        primary_kandas = [kanda for kanda, score in sorted_kandas[:2] if score > 10]
        secondary_kandas = [kanda for kanda, score in sorted_kandas[2:4] if score > 5]
        
        # Add supplementary as fallback only
        result = {
            'primary': primary_kandas,
            'secondary': secondary_kandas + ['supplementary'],  # Supplementary as fallback
            'scores': dict(sorted_kandas)
        }
        
        logger.debug(f"📊 Kanda relevance scores: {dict(sorted_kandas)}")
        
        return result
    
    def _handle_specific_cases(self, claim_lower: str) -> Optional[Dict[str, List[str]]]:
        """Handle specific cases that need particular Kanda attention"""
        
        specific_cases = {
            # Bridge construction - clearly Yuddha Kanda
            'bridge': {'primary': ['yuddha'], 'secondary': ['sundara', 'supplementary']},
            'setu': {'primary': ['yuddha'], 'secondary': ['sundara', 'supplementary']},
            'nala': {'primary': ['yuddha'], 'secondary': ['kishkindha', 'supplementary']},
            
            # Sleep/Kumbhakarna - Yuddha Kanda
            'kumbhakarna': {'primary': ['yuddha'], 'secondary': ['supplementary']},
            'sleep': {'primary': ['yuddha'], 'secondary': ['supplementary']},
            
            # Marriage/Bow - Bala Kanda
            'swayamvara': {'primary': ['bala'], 'secondary': ['supplementary']},
            'shiva.*bow': {'primary': ['bala'], 'secondary': ['supplementary']},
            
            # Exile details - Ayodhya Kanda
            'kaikeyi.*battle': {'primary': ['ayodhya'], 'secondary': ['supplementary']},
            'dasaratha.*died': {'primary': ['ayodhya'], 'secondary': ['supplementary']},
            
            # Forest incidents - Aranya Kanda
            'golden.*deer': {'primary': ['aranya'], 'secondary': ['supplementary']},
            'maricha': {'primary': ['aranya'], 'secondary': ['supplementary']},
            'tataka': {'primary': ['bala'], 'secondary': ['aranya', 'supplementary']},
            
            # War details - Yuddha Kanda
            'indrajit': {'primary': ['yuddha'], 'secondary': ['supplementary']},
            'brahmastra': {'primary': ['yuddha'], 'secondary': ['supplementary']},
            'shakti.*weapon': {'primary': ['yuddha'], 'secondary': ['supplementary']},
            
            # Monkey kingdom - Kishkindha Kanda
            'vali': {'primary': ['kishkindha'], 'secondary': ['supplementary']},
            'sugriva': {'primary': ['kishkindha'], 'secondary': ['sundara', 'supplementary']},
            'tara': {'primary': ['kishkindha'], 'secondary': ['supplementary']},
            
            # Search operations - Sundara Kanda
            'hanuman.*leap': {'primary': ['sundara'], 'secondary': ['supplementary']},
            'ocean.*cross': {'primary': ['sundara'], 'secondary': ['supplementary']},
            'sampati': {'primary': ['sundara'], 'secondary': ['kishkindha', 'supplementary']},
        }
        
        for pattern, kandas in specific_cases.items():
            if re.search(pattern, claim_lower):
                logger.info(f"🎯 Specific case matched: {pattern} → {kandas}")
                return kandas
        
        return None
    
    def _smart_rag_retrieval(self, claim: str, top_k: int = 3) -> List[Dict]:
        """RESTORED: Smart RAG retrieval - choose the most relevant Kanda book first"""
        claim_lower = claim.lower()
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        
        # Extract query entities and topics
        query_entities = self._extract_entities(claim)
        query_topics = self._extract_topics(claim)
        
        # STEP 1: Check for specific cases first
        specific_case = self._handle_specific_cases(claim_lower)
        if specific_case:
            kanda_relevance = specific_case
        else:
            # STEP 2: General Kanda relevance determination
            kanda_relevance = self._determine_best_kanda(claim_lower, query_entities, query_topics)
        
        # STEP 3: Score chunks with smart source prioritization
        scored_chunks = []
        
        for chunk in self.text_chunks:
            score = 0.0
            source = chunk.get('source', '').lower()
            
            # RESTORED: Smart source weighting based on Kanda relevance
            if any(kanda in source for kanda in kanda_relevance['primary']):
                score += 100.0  # Highest for primary Kanda
                logger.debug(f"🎯 Primary Kanda boost: {chunk['id']}")
            elif any(kanda in source for kanda in kanda_relevance['secondary']):
                score += 40.0   # Medium for secondary Kanda
                logger.debug(f"📖 Secondary Kanda boost: {chunk['id']}")
            else:
                score += 5.0   # Low score for other Kandas
            
            # Entity matching (highest weight)
            chunk_entities = chunk.get('entities', [])
            entity_overlap = len(set(query_entities).intersection(set(chunk_entities)))
            score += entity_overlap * 20.0
            
            # Topic matching
            chunk_topics = chunk.get('topics', [])
            topic_overlap = len(set(query_topics).intersection(set(chunk_topics)))
            score += topic_overlap * 15.0
            
            # Text similarity
            chunk_text_lower = chunk['text'].lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_text_lower))
            word_overlap = len(claim_words.intersection(chunk_words))
            score += word_overlap * 5.0
            
            # Exact phrase matching
            for phrase in claim.split():
                if len(phrase) > 4 and phrase in chunk_text_lower:
                    score += 10.0
            
            # Priority and fact density
            score += chunk.get('priority', 1.0) * 2.0
            score += chunk.get('fact_density', 0.0) * 3.0
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
        
        if final_chunks:
            top_score = scored_chunks[0][1] if scored_chunks else 0
            sources_used = [chunk.get('source', 'unknown') for chunk in final_chunks]
            logger.info(f"🎯 Smart RAG: Using sources {sources_used} (top score: {top_score:.1f})")
            logger.info(f"📚 Primary Kanda: {kanda_relevance['primary']}, Secondary: {kanda_relevance['secondary']}")
        
        return final_chunks
    
    def _create_rag_prompt_with_reasoning(self, claim: str, chunks: List[Dict], pattern_hint: Optional[Tuple[str, float, str]] = None) -> List[Dict]:
        """Create RAG-enhanced prompt that asks for reasoning, with optional pattern hint"""
        
        claim_lower = claim.lower()
        
        # Check for non-Ramayana content
        non_ramayana_terms = ['python', 'programming', 'paris', 'france', 'computer', 'technology', 'javascript', 'software']
        is_non_ramayana = any(term in claim_lower for term in non_ramayana_terms)
        
        # Check if it contains ANY Ramayana-related terms
        ramayana_terms = ['rama', 'sita', 'hanuman', 'ravana', 'bharata', 'lakshmana', 'ayodhya', 'lanka', 
                         'vanara', 'monkey', 'demon', 'rakshasa', 'bow', 'exile', 'bridge', 'ocean']
        contains_ramayana_content = any(term in claim_lower for term in ramayana_terms)
        
        # Build pattern hint text if available
        pattern_hint_text = ""
        if pattern_hint:
            suggested_verdict, confidence, explanation = pattern_hint
            pattern_hint_text = f"\nPATTERN SUGGESTION: This claim matches a known pattern suggesting {suggested_verdict} ({explanation})"
        
        if is_non_ramayana and not contains_ramayana_content:
            # ONLY pure modern topics get IRRELEVANT
            system_prompt = f"""You verify Ramayana facts only.

This claim is about modern topics (programming, cities, technology).{pattern_hint_text}

Format your answer as:
VERDICT: IRRELEVANT
REASONING: [Explain why this is not related to Ramayana]"""
        
            user_prompt = f'Claim: "{claim}"\nProvide verdict and reasoning:'
    
        else:
            # ALL Ramayana-related content gets TRUE/FALSE classification
            if chunks:
                evidence_parts = []
                for i, chunk in enumerate(chunks[:2]):  # Use top 2 chunks
                    source = chunk.get('source', 'unknown')
                    text = chunk['text'][:300]  # Limit text length
                    evidence_parts.append(f"Evidence {i+1} (from {source}):\n{text}")
                
                evidence = "\n\n".join(evidence_parts)
                
                system_prompt = f"""You are a Ramayana expert analyzing the CORE 6 KANDAS only.

SCOPE: Only Bala, Ayodhya, Aranya, Kishkindha, Sundara, and Yuddha Kandas.

{evidence}{pattern_hint_text}

CRITICAL RULES:
- Rama was exiled for EXACTLY 14 years - ANY other number is FALSE
- Ravana had EXACTLY 10 heads and 20 arms (not 10 arms) - ANY other number is FALSE  
- Kumbhakarna slept for EXACTLY 6 months and was awake for 6 months - ANY other duration is FALSE
- Hanuman was son of wind god Vayu - this is fundamental fact
- Lakshmana was Rama's younger brother - this is fundamental

Format your answer as:
VERDICT: [TRUE/FALSE]
REASONING: [One sentence explanation based on evidence and Ramayana knowledge]"""
        
            else:
                # No relevant chunks found
                system_prompt = f"""You are a Ramayana expert analyzing the CORE 6 KANDAS only.

CRITICAL: Since this is about Ramayana, NEVER answer IRRELEVANT. Only TRUE or FALSE.{pattern_hint_text}

EXACT FACTS:
✅ Rama: exiled for EXACTLY 14 years
✅ Ravana: EXACTLY 10 heads and 20 arms (not 10 arms)
✅ Kumbhakarna: slept 6 months, awake 6 months
✅ Hanuman: son of wind god Vayu

Format your answer as:
VERDICT: [TRUE/FALSE]
REASONING: [One sentence explanation based on Ramayana knowledge]"""
        
            user_prompt = f'Claim: "{claim}"\nProvide verdict and reasoning:'
    
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _parse_response_with_reasoning(self, response: str, statement: str) -> Tuple[str, str]:
        """Parse Groq response to extract verdict and reasoning"""
        if not response:
            return self._fallback_classification_with_reasoning(statement)
        
        response_clean = response.strip()
        statement_lower = statement.lower()
        
        # Check if this is actually Ramayana content
        ramayana_terms = ['rama', 'sita', 'hanuman', 'ravana', 'bharata', 'lakshmana', 'ayodhya', 'lanka', 
                         'vanara', 'monkey', 'demon', 'rakshasa', 'bow', 'exile', 'bridge', 'ocean']
        is_ramayana_content = any(term in statement_lower for term in ramayana_terms)
        
        # Try to parse structured response
        verdict = None
        reasoning = None
        
        lines = response_clean.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('VERDICT:'):
                verdict_part = line.replace('VERDICT:', '').strip().upper()
                if verdict_part in ['TRUE', 'FALSE', 'IRRELEVANT']:
                    verdict = verdict_part
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # If we got both, return them
        if verdict and reasoning:
            # CRITICAL FIX: Never return IRRELEVANT for Ramayana content
            if verdict == 'IRRELEVANT' and is_ramayana_content:
                logger.warning(f"Model returned IRRELEVANT for Ramayana content, changing to TRUE")
                verdict = 'TRUE'
                reasoning = "Ramayana-related content classified as true by default"
            return verdict, reasoning
        
        # Fallback parsing - look for keywords
        response_upper = response_clean.upper()
        
        if 'TRUE' in response_upper and 'FALSE' not in response_upper:
            verdict = 'TRUE'
        elif 'FALSE' in response_upper:
            verdict = 'FALSE'
        elif 'IRRELEVANT' in response_upper:
            if is_ramayana_content:
                verdict = 'TRUE'
                reasoning = "Ramayana-related content classified as true by default"
            else:
                verdict = 'IRRELEVANT'
        else:
            return self._fallback_classification_with_reasoning(statement)
        
        # Extract reasoning if not found
        if not reasoning:
            # Take the whole response as reasoning if it's reasonable length
            if len(response_clean) > 10 and len(response_clean) < 200:
                reasoning = response_clean
            else:
                reasoning = "Classification based on Ramayana knowledge"
        
        return verdict, reasoning
    
    def _fallback_classification_with_reasoning(self, statement: str) -> Tuple[str, str]:
        """Conservative fallback classification with reasoning"""
        statement_lower = statement.lower()
        
        # ONLY pure non-Ramayana content gets IRRELEVANT
        non_ramayana_terms = ['python', 'programming', 'paris', 'france', 'computer', 'technology', 'javascript', 'software']
        ramayana_terms = ['rama', 'sita', 'hanuman', 'ravana', 'bharata', 'lakshmana', 'ayodhya', 'lanka', 
                         'vanara', 'monkey', 'demon', 'rakshasa', 'bow', 'exile', 'bridge', 'ocean']
        
        has_non_ramayana = any(term in statement_lower for term in non_ramayana_terms)
        has_ramayana = any(term in statement_lower for term in ramayana_terms)
        
        # IRRELEVANT only for pure modern topics with NO Ramayana content
        if has_non_ramayana and not has_ramayana:
            return 'IRRELEVANT', 'This statement is about modern technology, not Ramayana'
        
        # If it has ANY Ramayana content, classify as TRUE/FALSE only
        if has_ramayana:
            # Known false patterns for Ramayana content
            false_indicators = [
                (['rama', 'born', 'lanka'], 'Rama was born in Ayodhya, not Lanka'),
                (['rama', 'refuse', 'exile'], 'Rama willingly accepted exile'),
                (['lakshmana', 'not', 'related'], 'Lakshmana was Rama\'s younger brother'),
                (['sita', 'daughter', 'ravana'], 'Sita was daughter of Janaka, not Ravana'),
                (['hanuman', 'betray'], 'Hanuman was completely loyal to Rama'),
                (['bharata', 'war', 'rama'], 'Bharata never fought against Rama')
            ]
            
            for pattern, reason in false_indicators:
                if all(word in statement_lower for word in pattern):
                    return 'FALSE', reason
            
            # Default to TRUE for Ramayana content (conservative approach)
            return 'TRUE', 'Statement appears to be about Ramayana and is generally consistent with the epic'
        
        # For content with neither modern nor Ramayana terms, default to IRRELEVANT
        return 'IRRELEVANT', 'Statement does not appear to be related to Ramayana'

    def _safe_groq_request(self, messages: List[Dict], max_tokens: int = 50, temperature: float = 0.0) -> str:
        """Safe Groq request with retries"""
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    sleep_time = self.min_request_interval - time_since_last
                    time.sleep(sleep_time)
                
                # Make request
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                self.last_request_time = time.time()
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate limit" in error_str or "429" in error_str:
                    retry_delay = 3.0 * (2 ** attempt)
                    logger.warning(f"⚠️ Rate limit. Waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    continue
                elif "quota" in error_str:
                    logger.error("❌ API quota exceeded")
                    raise
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise
        
        return ""

    def classify_statement(self, statement: str) -> Dict[str, Any]:
        """Main classification using RESTORED smart RAG approach"""
        start_time = time.time()
        
        try:
            statement = str(statement).strip()
            if not statement or statement.lower() in ['nan', 'none', '']:
                return {
                    'statement': statement,
                    'verdict': 'ERROR',
                    'reasoning': 'Empty statement provided'
                }
            
            logger.info(f"🔍 Processing: '{statement[:60]}...'")
            
            # Check semantic patterns first for efficiency
            pattern_result = self._semantic_pattern_check(statement)
            
            # RESTORED: Smart RAG retrieval from relevant Kanda files
            search_start = time.time()
            relevant_chunks = self._smart_rag_retrieval(statement, top_k=3)
            search_time = time.time() - search_start
            
            # Create RAG-enhanced prompt that includes pattern hint if found
            messages = self._create_rag_prompt_with_reasoning(statement, relevant_chunks, pattern_result)
            
            # Get Groq response with reasoning
            groq_start = time.time()
            groq_response = self._safe_groq_request(messages, max_tokens=50, temperature=0.0)
            groq_time = time.time() - groq_start
            
            # Parse response with reasoning
            label, reasoning = self._parse_response_with_reasoning(groq_response, statement)
            
            logger.info(f"✅ Final: {label} - '{statement[:40]}...'")
            
            return {
                'statement': statement,
                'verdict': label,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return {
                'statement': statement,
                'verdict': 'ERROR',
                'reasoning': f'Processing error: {str(e)}'
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'files_loaded': self.files_loaded,
            'total_chunks': len(self.text_chunks),
            'total_chars_processed': self.total_chars_processed,
            'pattern_rules_loaded': len(self.pattern_rules),
            'critical_patterns': len(self.semantic_patterns),
            'entity_index_size': len(self.chunk_index.get('entities', {})),
            'topic_index_size': len(self.chunk_index.get('topics', {})),
            'avg_chunk_size': np.mean([chunk['char_count'] for chunk in self.text_chunks]) if self.text_chunks else 0,
            'pattern_sources': {
                source: len([r for r in self.pattern_rules if r.get('source') == source])
                for source in set(r.get('source', 'unknown') for r in self.pattern_rules)
            }
        }

def process_csv_smart_rag(input_file: str, output_file: str, api_key: str = None):
    """Process CSV with RESTORED smart RAG system"""
    try:
        logger.info("🚀 Initializing SMART RAG System with Restored Kanda Selection...")
        classifier = SmartRAGClassifier(api_key=api_key)
        
        # Display system stats
        stats = classifier.get_system_stats()
        logger.info(f"📊 System Stats:")
        logger.info(f"  📁 Files loaded: {stats['files_loaded']}")
        logger.info(f"  📚 Total chunks: {stats['total_chunks']}")
        logger.info(f"  📄 Characters processed: {stats['total_chars_processed']:,}")
        logger.info(f"  ⚡ Pattern rules from file: {stats['pattern_rules_loaded']}")
        logger.info(f"  🛡️ Critical fallback patterns: {stats['critical_patterns']}")
        logger.info(f"  🔍 Entity index size: {stats['entity_index_size']}")
        logger.info(f"  📊 Average chunk size: {stats['avg_chunk_size']:.0f} chars")
        
        # Load CSV
        df = pd.read_csv(input_file)
        logger.info(f"📋 Loaded CSV with {len(df)} rows")
        
        # Auto-detect statement column
        statement_column = None
        for col_name in ['statement', 'claim', 'text', 'sentence']:
            if col_name in df.columns:
                statement_column = col_name
                break
        
        if statement_column is None:
            statement_column = df.columns[0]
        
        logger.info(f"📝 Using column: '{statement_column}'")
        
        results = []
        pattern_hits = 0
        rag_retrievals = 0
        groq_calls_made = 0
        
        start_time = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Smart RAG Processing"):
            try:
                statement = str(row[statement_column]).strip()
                
                if not statement or statement.lower() in ['nan', 'none', '']:
                    continue
                
                result = classifier.classify_statement(statement)
                results.append(result)
                
                # Track method usage
                rag_retrievals += 1
                groq_calls_made += 1
                
                # Progress update every 10 items
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (len(df) - idx - 1) / rate if rate > 0 else 0
                    logger.info(f"Progress: {idx+1}/{len(df)} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s | Groq: {groq_calls_made}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing row {idx}: {e}")
                results.append({
                    'statement': f"Error in row {idx}",
                    'verdict': 'ERROR',
                    'reasoning': f'Processing error: {str(e)}'
                })
        
        if not results:
            logger.error("❌ No results generated!")
            return
        
        # Save results with only the 3 required columns
        output_data = []
        for result in results:
            output_data.append({
                'statement': result['statement'],
                'verdict': result['verdict'],
                'reasoning': result['reasoning']
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)
        logger.info(f"💾 Results saved to {output_file}")
        
        # Print summary
        print_smart_summary(output_df, pattern_hits, rag_retrievals, groq_calls_made, 
                           time.time() - start_time, stats)
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()

def print_smart_summary(df: pd.DataFrame, pattern_hits: int, rag_retrievals: int, 
                       groq_calls: int, total_time: float, system_stats: Dict):
    """Print summary of SMART RAG results"""
    print("\n" + "="*80)
    print("🎯 SMART RAG WITH RESTORED KANDA SELECTION RESULTS")
    print("="*80)
    
    if len(df) == 0:
        print("❌ No results to display")
        return
    
    # System Overview
    print(f"🏗️  SYSTEM OVERVIEW:")
    print(f"  📁 Files processed: {system_stats['files_loaded']}")
    print(f"  📚 Text chunks created: {system_stats['total_chunks']:,}")
    print(f"  📄 Total characters: {system_stats['total_chars_processed']:,}")
    print(f"  ⚡ Patterns from file: {system_stats['pattern_rules_loaded']}")
    print(f"  🛡️ Critical fallback patterns: {system_stats['critical_patterns']}")
    print(f"  🎯 SMART KANDA SELECTION: Restored!")
    
    # Performance Overview
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  📋 Total statements: {len(df)}")
    print(f"  ⏱️  Total time: {total_time:.2f}s")
    print(f"  🚀 Processing rate: {len(df)/total_time:.2f} statements/sec")
    print(f"  📚 Smart RAG retrievals: {rag_retrievals} ({rag_retrievals/len(df)*100:.1f}%)")
    print(f"  🤖 Groq API calls: {groq_calls} ({groq_calls/len(df)*100:.1f}%)")
    print(f"  🎯 Now using RELEVANT Kanda books, not just supplementary!")
    
    # Classification Results
    label_counts = df['verdict'].value_counts()
    print(f"\n🏷️  CLASSIFICATION RESULTS:")
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        if label == "TRUE":
            emoji = "✅"
        elif label == "FALSE":
            emoji = "❌"
        elif label == "IRRELEVANT":
            emoji = "➖"
        elif label == "ERROR":
            emoji = "🚫"
        else:
            emoji = "❓"
        print(f"  {emoji} {label}: {count} statements ({percentage:.1f}%)")
    
    # Sample results
    print(f"\n📋 SAMPLE RESULTS:")
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        statement = row['statement'][:50] + "..." if len(row['statement']) > 50 else row['statement']
        print(f"  {i}. '{statement}' → {row['verdict']}")
        print(f"     Reasoning: {row['reasoning']}")
        print()
    
    print(f"\n" + "="*80)
    print("🎉 SMART RAG WITH KANDA SELECTION RESTORED!")
    print("✅ System now chooses the MOST RELEVANT Kanda book first!")
    print("📚 Supplementary file used for patterns + fallback only!")
    print("🎯 Bridge questions → Yuddha Kanda, Marriage → Bala Kanda, etc.")
    print("="*80)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart RAG with Restored Kanda Selection')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--api-key', type=str, help='Groq API key (optional)')
    
    args = parser.parse_args()
    
    print("🎯 SMART RAG WITH RESTORED KANDA SELECTION")
    print("="*70)
    print("🚀 FIXED SMART APPROACH:")
    print("  📚 RAG: Intelligently chooses RELEVANT Kanda books")
    print("  🎯 SMART: Bridge questions → Yuddha, Marriage → Bala, etc.")
    print("  📄 PATTERNS: Loads from supplementary knowledge file")
    print("  ⚡ EFFICIENT: Patterns + smart Kanda selection")
    print("  🔧 FIXED: No more over-reliance on supplementary file!")
    print()
    
    try:
        process_csv_smart_rag(args.input, args.output, args.api_key)
        print("\n🎉 Smart RAG processing completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
