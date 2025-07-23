import pytest
import time
from unittest.mock import Mock, patch

from self_healing_router import SelfHealingRouter
from semantic_rewriter import SemanticTaskRewriter
from multimodal_embeddings import MultimodalMemoryEmbeddings
from pipeline_manager import ModelPipelineManager
from permission_manager import PermissionManager, PermissionTier
from culture_adapter import CultureAwareAdapter
from task_decomposer import TaskDecomposer
from outcome_scorer import OutcomeBasedScorer
from agent_collaboration import CrossAgentCollaborator
from emotion_processor import VoiceEmotionProcessor
from zkpe_processor import ZKPEProcessor

class TestSelfHealingRouter:
    def test_router_initialization(self):
        router = SelfHealingRouter()
        assert router.response_cache == {}
        assert router.model_health == {}
        assert router.circuit_breakers == {}
    
    def test_cache_key_generation(self):
        router = SelfHealingRouter()
        key1 = router._generate_cache_key("test prompt", "gpt-4o", "text")
        key2 = router._generate_cache_key("test prompt", "gpt-4o", "text")
        key3 = router._generate_cache_key("different prompt", "gpt-4o", "text")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_health_stats(self):
        router = SelfHealingRouter()
        stats = router.get_health_stats()
        
        assert 'model_health' in stats
        assert 'circuit_breakers' in stats
        assert 'cache_size' in stats

class TestSemanticRewriter:
    def test_ambiguity_detection(self):
        rewriter = SemanticTaskRewriter()
        
        assert rewriter._is_ambiguous("write something")
        assert rewriter._is_ambiguous("help me")
        assert rewriter._is_ambiguous("do this")
        
        assert not rewriter._is_ambiguous("Write a Python function to calculate fibonacci numbers")
        assert not rewriter._is_ambiguous("Translate 'hello world' to Spanish")
    
    def test_ambiguity_analysis(self):
        rewriter = SemanticTaskRewriter()
        analysis = rewriter.analyze_ambiguity_level("write something good")
        
        assert 'is_ambiguous' in analysis
        assert 'ambiguity_score' in analysis
        assert 'ambiguity_types' in analysis

class TestMultimodalEmbeddings:
    def test_embedding_initialization(self):
        embeddings = MultimodalMemoryEmbeddings()
        assert embeddings.embedding_dim == 384
        assert embeddings.memory_store == {}
    
    def test_embedding_stats(self):
        embeddings = MultimodalMemoryEmbeddings()
        stats = embeddings.get_embedding_stats()
        
        assert 'total_embeddings' in stats
        assert 'by_content_type' in stats
        assert 'has_faiss' in stats

class TestPipelineManager:
    def test_pipeline_initialization(self):
        manager = ModelPipelineManager()
        assert 'image_analysis_summary' in manager.predefined_pipelines
        assert 'code_review_improve' in manager.predefined_pipelines
    
    def test_complexity_detection(self):
        manager = ModelPipelineManager()
        
        assert manager._is_complex_multi_step_task("analyze this image and then summarize the findings")
        
        assert not manager._is_complex_multi_step_task("hello world")
    
    def test_pipeline_info(self):
        manager = ModelPipelineManager()
        info = manager.get_pipeline_info('image_analysis_summary')
        
        assert info is not None
        assert 'steps' in info
        assert 'step_details' in info

class TestPermissionManager:
    def test_permission_tiers(self):
        manager = PermissionManager()
        
        free_info = manager.get_tier_info(PermissionTier.FREE)
        assert 'models' in free_info
        assert 'daily_requests' in free_info
        
        pro_info = manager.get_tier_info(PermissionTier.PRO)
        assert len(pro_info['models']) > len(free_info['models'])
    
    def test_permission_checking(self):
        manager = PermissionManager()
        
        granted, msg = manager.check_permission('test_user', 'gpt-4o-mini', 'text')
        assert granted
        
        granted, msg = manager.check_permission('test_user', 'gpt-4o', 'text')
        assert not granted
    
    def test_usage_tracking(self):
        manager = PermissionManager()
        manager.track_usage('test_user', 'gpt-4o-mini', 'text', True)
        
        usage = manager.get_user_usage('test_user')
        assert usage['total_requests'] >= 1

class TestCultureAdapter:
    def test_cultural_contexts(self):
        adapter = CultureAwareAdapter()
        
        languages = adapter.get_supported_languages()
        assert 'en' in languages
        assert 'ja' in languages
        assert 'de' in languages
    
    def test_context_detection(self):
        adapter = CultureAwareAdapter()
        
        context = adapter.detect_context_type("create a business proposal for our company")
        assert context == 'business_context'
        
        context = adapter.detect_context_type("write a creative story about dragons")
        assert context == 'creative_context'
    
    def test_cultural_analysis(self):
        adapter = CultureAwareAdapter()
        analysis = adapter.analyze_cultural_needs("こんにちは", "test_user")
        
        assert 'detected_language' in analysis
        assert 'needs_adaptation' in analysis

class TestTaskDecomposer:
    def test_complexity_analysis(self):
        decomposer = TaskDecomposer()
        
        analysis = decomposer.analyze_task_complexity("build a complete web application with frontend and backend")
        assert analysis['complexity_level'] in ['medium', 'high']
        assert analysis['should_decompose']
        
        analysis = decomposer.analyze_task_complexity("hello")
        assert analysis['complexity_level'] == 'low'
        assert not analysis['should_decompose']
    
    def test_complexity_detection(self):
        decomposer = TaskDecomposer()
        
        assert decomposer._is_complex_task("build a complete solution")
        assert decomposer._is_complex_task("create and then analyze and then summarize")
        
        assert not decomposer._is_complex_task("hello world")

class TestOutcomeScorer:
    def test_scorer_initialization(self):
        scorer = OutcomeBasedScorer()
        assert 'user_feedback' in scorer.outcome_weights
        assert 'task_completion' in scorer.outcome_weights
    
    def test_quality_scoring(self):
        scorer = OutcomeBasedScorer()
        
        score = scorer._calculate_quality_score(100, 300, True)
        assert 0.5 <= score <= 1.0
        
        score = scorer._calculate_quality_score(100, 300, False)
        assert score == 0.0
    
    def test_efficiency_scoring(self):
        scorer = OutcomeBasedScorer()
        
        score = scorer._calculate_efficiency_score(2.0, 'text')
        assert score >= 0.8
        
        score = scorer._calculate_efficiency_score(30.0, 'text')
        assert score <= 0.5

class TestAgentCollaborator:
    def test_collaboration_initialization(self):
        collaborator = CrossAgentCollaborator()
        assert 'text_specialist' in collaborator.agent_capabilities
        assert 'code_specialist' in collaborator.agent_capabilities
    
    def test_collaboration_analysis(self):
        collaborator = CrossAgentCollaborator()
        
        analysis = collaborator._analyze_collaboration_need("create code and design a user interface")
        assert analysis['needs_collaboration']
        
        analysis = collaborator._analyze_collaboration_need("hello world")
        assert not analysis['needs_collaboration']
    
    def test_collaboration_stats(self):
        collaborator = CrossAgentCollaborator()
        stats = collaborator.get_collaboration_stats()
        
        assert 'available_agents' in stats
        assert 'agent_capabilities' in stats

class TestEmotionProcessor:
    def test_emotion_patterns(self):
        processor = VoiceEmotionProcessor()
        assert 'urgent' in processor.emotion_patterns
        assert 'frustrated' in processor.emotion_patterns
        assert 'excited' in processor.emotion_patterns
    
    def test_emotion_detection(self):
        processor = VoiceEmotionProcessor()
        
        emotions = processor._detect_emotions("This is urgent! Please help immediately!")
        emotion_names = [e['emotion'] for e in emotions]
        assert 'urgent' in emotion_names
        
        emotions = processor._detect_emotions("This is amazing and fantastic!")
        emotion_names = [e['emotion'] for e in emotions]
        assert 'excited' in emotion_names
    
    def test_urgency_detection(self):
        processor = VoiceEmotionProcessor()
        
        urgency = processor._detect_urgency("urgent emergency help needed now")
        assert urgency['level'] == 'high'
        
        urgency = processor._detect_urgency("hello how are you")
        assert urgency['level'] == 'normal'
    
    def test_emotion_stats(self):
        processor = VoiceEmotionProcessor()
        stats = processor.get_emotion_stats()
        
        assert 'supported_emotions' in stats
        assert 'response_styles' in stats

class TestZKPEProcessor:
    def test_zkpe_initialization(self):
        processor = ZKPEProcessor()
        assert processor.sensitive_patterns is not None
        assert 'credit_card' in processor.sensitive_patterns
        assert 'email' in processor.sensitive_patterns
    
    def test_sensitivity_analysis(self):
        processor = ZKPEProcessor()
        
        analysis = processor._analyze_sensitivity("My email is test@example.com and my phone is 555-123-4567")
        assert analysis['is_sensitive']
        assert 'email' in analysis['detected_types']
        
        analysis = processor._analyze_sensitivity("Hello, how are you today?")
        assert not analysis['is_sensitive']
    
    def test_privacy_stats(self):
        processor = ZKPEProcessor()
        stats = processor.get_privacy_stats()
        
        assert 'supported_patterns' in stats
        assert 'privacy_levels' in stats

class TestIntegration:
    def test_all_modules_importable(self):
        """Test that all modules can be imported without errors"""
        assert True
    
    def test_module_compatibility(self):
        """Test that modules can work together"""
        router = SelfHealingRouter()
        rewriter = SemanticTaskRewriter()
        embeddings = MultimodalMemoryEmbeddings()
        
        assert router is not None
        assert rewriter is not None
        assert embeddings is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
