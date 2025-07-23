import re
from typing import Dict, Tuple, List, Optional, Any
import logging

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available, using fallback sentiment analysis")

class VoiceEmotionProcessor:
    def __init__(self):
        self.emotion_patterns = {
            'urgent': {
                'keywords': ['urgent', 'asap', 'immediately', 'quickly', 'rush', 'emergency', 'critical'],
                'punctuation': ['!!!', '!!', '!'],
                'caps_threshold': 0.3
            },
            'frustrated': {
                'keywords': ['frustrated', 'annoyed', 'stuck', 'not working', 'broken', 'terrible', 'awful', 'hate'],
                'punctuation': ['!!!', '??', '!?'],
                'caps_threshold': 0.2
            },
            'excited': {
                'keywords': ['excited', 'amazing', 'awesome', 'love', 'fantastic', 'wonderful', 'great', 'brilliant'],
                'punctuation': ['!', '!!', '!!!'],
                'caps_threshold': 0.1
            },
            'confused': {
                'keywords': ['confused', "don't understand", 'unclear', 'help', 'lost', 'puzzled', 'baffled'],
                'punctuation': ['???', '??', '?'],
                'caps_threshold': 0.0
            },
            'formal': {
                'keywords': ['please', 'kindly', 'would you', 'could you', 'thank you', 'appreciate', 'grateful'],
                'punctuation': ['.', ';'],
                'caps_threshold': 0.0
            },
            'angry': {
                'keywords': ['angry', 'mad', 'furious', 'outraged', 'livid', 'pissed', 'damn', 'stupid'],
                'punctuation': ['!!!', '!!!!'],
                'caps_threshold': 0.4
            },
            'sad': {
                'keywords': ['sad', 'depressed', 'down', 'upset', 'disappointed', 'heartbroken', 'miserable'],
                'punctuation': ['...', ':(', ':-('],
                'caps_threshold': 0.0
            },
            'happy': {
                'keywords': ['happy', 'joy', 'delighted', 'pleased', 'cheerful', 'glad', 'thrilled'],
                'punctuation': [':)', ':-)', ':D', '😊', '😄'],
                'caps_threshold': 0.1
            }
        }
        
        self.intensity_modifiers = {
            'very': 1.5,
            'extremely': 2.0,
            'really': 1.3,
            'quite': 1.2,
            'somewhat': 0.8,
            'slightly': 0.6,
            'a bit': 0.7,
            'totally': 1.8,
            'absolutely': 1.9
        }
        
        self.response_styles = {
            'urgent': {
                'tone': 'immediate and focused',
                'prefix': "I'll help you right away.",
                'structure': 'direct and actionable'
            },
            'frustrated': {
                'tone': 'supportive and understanding',
                'prefix': "I understand this can be frustrating. Let me help:",
                'structure': 'empathetic and solution-focused'
            },
            'excited': {
                'tone': 'enthusiastic and engaging',
                'prefix': "Great question! I'm excited to help with this.",
                'structure': 'energetic and detailed'
            },
            'confused': {
                'tone': 'patient and explanatory',
                'prefix': "Let me break this down clearly for you.",
                'structure': 'step-by-step and clear'
            },
            'formal': {
                'tone': 'professional and respectful',
                'prefix': "Thank you for your inquiry.",
                'structure': 'structured and comprehensive'
            },
            'angry': {
                'tone': 'calm and de-escalating',
                'prefix': "I understand you're upset. Let me help resolve this.",
                'structure': 'calm and solution-oriented'
            },
            'sad': {
                'tone': 'gentle and supportive',
                'prefix': "I'm here to help and support you.",
                'structure': 'compassionate and encouraging'
            },
            'happy': {
                'tone': 'positive and upbeat',
                'prefix': "Wonderful! I'm happy to help with this.",
                'structure': 'positive and engaging'
            }
        }
        
    def analyze_emotion_and_tone(self, text: str) -> Dict[str, Any]:
        """Analyze emotion, tone, and sentiment from text input."""
        try:
            sentiment = self._analyze_sentiment(text)
            
            detected_emotions = self._detect_emotions(text)
            
            urgency_level = self._detect_urgency(text)
            
            tone = self._analyze_tone(text, sentiment['polarity'])
            
            intensity = self._analyze_intensity(text, detected_emotions)
            
            confidence = self._calculate_confidence(detected_emotions, sentiment, urgency_level)
            
            return {
                'sentiment': sentiment,
                'emotions': detected_emotions,
                'urgency': urgency_level,
                'tone': tone,
                'intensity': intensity,
                'confidence': confidence,
                'response_style': self._recommend_response_style(detected_emotions, urgency_level, sentiment)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing emotion and tone: {e}")
            return self._get_default_analysis()
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob or fallback method."""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            else:
                return self._fallback_sentiment_analysis(text)
                
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis when TextBlob is not available."""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
            'frustrated', 'disappointed', 'sad', 'upset', 'broken', 'wrong'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return {'polarity': 0.0, 'subjectivity': 0.5}
        
        polarity = (positive_count - negative_count) / max(total_words, 1)
        polarity = max(-1, min(1, polarity * 5))  # Scale and clamp
        
        subjective_indicators = positive_count + negative_count
        subjectivity = min(1, subjective_indicators / max(total_words, 1) * 3)
        
        return {'polarity': polarity, 'subjectivity': subjectivity}
    
    def _detect_emotions(self, text: str) -> List[Dict[str, Any]]:
        """Detect emotions in text based on patterns."""
        detected_emotions = []
        text_lower = text.lower()
        
        for emotion, patterns in self.emotion_patterns.items():
            emotion_score = 0
            detected_indicators = []
            
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    emotion_score += 1
                    detected_indicators.append(f"keyword: {keyword}")
            
            for punct in patterns['punctuation']:
                if punct in text:
                    emotion_score += 0.5
                    detected_indicators.append(f"punctuation: {punct}")
            
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            if caps_ratio >= patterns['caps_threshold']:
                emotion_score += caps_ratio
                detected_indicators.append(f"caps_ratio: {caps_ratio:.2f}")
            
            if emotion_score > 0:
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': min(1.0, emotion_score / 3),  # Normalize to 0-1
                    'indicators': detected_indicators
                })
        
        detected_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_emotions
    
    def _detect_urgency(self, text: str) -> Dict[str, Any]:
        """Detect urgency level in text."""
        urgency_indicators = {
            'high': ['urgent', 'emergency', 'critical', 'asap', 'immediately', 'now'],
            'medium': ['soon', 'quickly', 'fast', 'rush', 'hurry'],
            'low': ['when possible', 'eventually', 'sometime', 'later']
        }
        
        text_lower = text.lower()
        urgency_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    urgency_scores[level] += 1
        
        if urgency_scores['high'] > 0:
            level = 'high'
            score = min(1.0, urgency_scores['high'] / 2)
        elif urgency_scores['medium'] > 0:
            level = 'medium'
            score = min(1.0, urgency_scores['medium'] / 3)
        elif urgency_scores['low'] > 0:
            level = 'low'
            score = min(1.0, urgency_scores['low'] / 2)
        else:
            level = 'normal'
            score = 0.0
        
        return {
            'level': level,
            'score': score,
            'indicators': urgency_scores
        }
    
    def _analyze_tone(self, text: str, sentiment_polarity: float) -> Dict[str, Any]:
        """Analyze overall tone of the text."""
        tone_indicators = {
            'professional': ['please', 'thank you', 'kindly', 'appreciate', 'regards'],
            'casual': ['hey', 'hi', 'thanks', 'cool', 'awesome', 'yeah'],
            'technical': ['implement', 'configure', 'optimize', 'debug', 'analyze'],
            'emotional': ['feel', 'think', 'believe', 'hope', 'wish', 'love', 'hate']
        }
        
        text_lower = text.lower()
        tone_scores = {}
        
        for tone, indicators in tone_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                tone_scores[tone] = score
        
        if tone_scores:
            primary_tone = max(tone_scores, key=tone_scores.get)
            confidence = min(1.0, tone_scores[primary_tone] / 3)
        else:
            primary_tone = 'neutral'
            confidence = 0.5
        
        if sentiment_polarity > 0.3:
            tone_modifier = 'positive'
        elif sentiment_polarity < -0.3:
            tone_modifier = 'negative'
        else:
            tone_modifier = 'neutral'
        
        return {
            'primary_tone': primary_tone,
            'tone_modifier': tone_modifier,
            'confidence': confidence,
            'tone_scores': tone_scores
        }
    
    def _analyze_intensity(self, text: str, detected_emotions: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional intensity."""
        text_lower = text.lower()
        intensity_score = 0
        modifiers_found = []
        
        for modifier, multiplier in self.intensity_modifiers.items():
            if modifier in text_lower:
                intensity_score += multiplier - 1  # Subtract 1 to get the boost
                modifiers_found.append(modifier)
        
        words = text_lower.split()
        repeated_words = [word for word in set(words) if words.count(word) > 1]
        if repeated_words:
            intensity_score += len(repeated_words) * 0.2
        
        exclamation_count = text.count('!')
        if exclamation_count > 1:
            intensity_score += exclamation_count * 0.3
        
        if detected_emotions:
            max_emotion_confidence = max(e['confidence'] for e in detected_emotions)
            intensity_score += max_emotion_confidence
        
        intensity_level = min(1.0, intensity_score)
        
        if intensity_level >= 0.8:
            level_name = 'very_high'
        elif intensity_level >= 0.6:
            level_name = 'high'
        elif intensity_level >= 0.4:
            level_name = 'medium'
        elif intensity_level >= 0.2:
            level_name = 'low'
        else:
            level_name = 'very_low'
        
        return {
            'level': level_name,
            'score': intensity_level,
            'modifiers_found': modifiers_found,
            'repeated_words': repeated_words
        }
    
    def _calculate_confidence(self, emotions: List[Dict], sentiment: Dict, urgency: Dict) -> float:
        """Calculate overall confidence in emotion analysis."""
        confidence_factors = []
        
        if emotions:
            avg_emotion_confidence = sum(e['confidence'] for e in emotions) / len(emotions)
            confidence_factors.append(avg_emotion_confidence)
        
        sentiment_confidence = sentiment.get('subjectivity', 0.5)
        confidence_factors.append(sentiment_confidence)
        
        urgency_confidence = urgency.get('score', 0.0)
        confidence_factors.append(urgency_confidence)
        
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5
    
    def _recommend_response_style(self, emotions: List[Dict], urgency: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Recommend response style based on analysis."""
        primary_emotion = emotions[0]['emotion'] if emotions else 'neutral'
        
        if urgency['level'] == 'high':
            primary_emotion = 'urgent'
        
        style_info = self.response_styles.get(primary_emotion, {
            'tone': 'neutral and helpful',
            'prefix': "I'm here to help.",
            'structure': 'clear and informative'
        })
        
        return {
            'primary_emotion': primary_emotion,
            'recommended_tone': style_info['tone'],
            'suggested_prefix': style_info['prefix'],
            'response_structure': style_info['structure'],
            'urgency_level': urgency['level']
        }
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when processing fails."""
        return {
            'sentiment': {'polarity': 0.0, 'subjectivity': 0.5},
            'emotions': [],
            'urgency': {'level': 'normal', 'score': 0.0, 'indicators': {}},
            'tone': {'primary_tone': 'neutral', 'tone_modifier': 'neutral', 'confidence': 0.5},
            'intensity': {'level': 'low', 'score': 0.0, 'modifiers_found': []},
            'confidence': 0.5,
            'response_style': {
                'primary_emotion': 'neutral',
                'recommended_tone': 'neutral and helpful',
                'suggested_prefix': "I'm here to help.",
                'response_structure': 'clear and informative',
                'urgency_level': 'normal'
            }
        }
    
    def adapt_response_to_emotion(self, response: str, emotion_analysis: Dict) -> str:
        """Adapt response based on emotional analysis."""
        try:
            style = emotion_analysis.get('response_style', {})
            prefix = style.get('suggested_prefix', '')
            
            if prefix and not response.startswith(prefix):
                adapted_response = f"{prefix} {response}"
            else:
                adapted_response = response
            
            return adapted_response
            
        except Exception as e:
            logging.error(f"Error adapting response to emotion: {e}")
            return response
    
    def get_emotion_stats(self) -> Dict[str, Any]:
        """Get statistics about emotion processing capabilities."""
        return {
            'supported_emotions': list(self.emotion_patterns.keys()),
            'intensity_modifiers': len(self.intensity_modifiers),
            'response_styles': len(self.response_styles),
            'textblob_available': TEXTBLOB_AVAILABLE
        }
