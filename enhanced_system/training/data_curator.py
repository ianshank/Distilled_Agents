"""
Data Curation Module
Provides data quality filtering, augmentation, and balancing
"""

import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import Counter
import random


logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Single data sample"""
    id: str
    prompt: str
    completion: str
    category: Optional[str] = None
    quality_score: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataCurator:
    """
    Curate and enhance training data
    
    Features:
    - Quality filtering
    - Data augmentation (paraphrasing)
    - Dataset balancing
    - Diversity analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data curator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.enable_augmentation = config.get('enable_augmentation', True)
        self.augmentation_factor = config.get('augmentation_factor', 2)
        self.enable_balancing = config.get('enable_balancing', True)
        self.diversity_threshold = config.get('diversity_threshold', 0.8)
        
        logger.info(
            f"DataCurator initialized (quality_threshold: {self.quality_threshold})"
        )
    
    def curate_dataset(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Curate complete dataset
        
        Args:
            data: List of data samples
        
        Returns:
            Curated dataset
        """
        logger.info(f"Curating dataset with {len(data)} samples")
        
        # Convert to DataSample objects
        samples = self._parse_samples(data)
        
        # Calculate quality scores
        samples = self._calculate_quality_scores(samples)
        
        # Filter by quality
        filtered = self.filter_quality(samples)
        logger.info(f"After quality filtering: {len(filtered)} samples")
        
        # Categorize
        categorized = self._categorize_samples(filtered)
        
        # Balance dataset
        if self.enable_balancing:
            balanced = self.balance_dataset(categorized)
            logger.info(f"After balancing: {len(balanced)} samples")
        else:
            balanced = categorized
        
        # Augment data
        if self.enable_augmentation:
            augmented = self.augment_data(balanced)
            logger.info(f"After augmentation: {len(augmented)} samples")
        else:
            augmented = balanced
        
        # Check diversity
        diversity_score = self._calculate_diversity(augmented)
        logger.info(f"Dataset diversity score: {diversity_score:.3f}")
        
        # Convert back to dictionaries
        curated_data = [self._sample_to_dict(sample) for sample in augmented]
        
        return curated_data
    
    def _parse_samples(self, data: List[Dict[str, Any]]) -> List[DataSample]:
        """Parse raw data into DataSample objects"""
        samples = []
        
        for i, item in enumerate(data):
            sample = DataSample(
                id=item.get('id', f"sample_{i}"),
                prompt=item.get('prompt', item.get('input', '')),
                completion=item.get('completion', item.get('output', '')),
                category=item.get('category'),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        return samples
    
    def _calculate_quality_scores(
        self,
        samples: List[DataSample]
    ) -> List[DataSample]:
        """Calculate quality scores for samples"""
        for sample in samples:
            sample.quality_score = self._assess_quality(sample)
        
        return samples
    
    def _assess_quality(self, sample: DataSample) -> float:
        """
        Assess quality of a single sample
        
        Args:
            sample: Data sample
        
        Returns:
            Quality score (0-1)
        """
        score = 0.5  # Base score
        
        # Check prompt quality
        if len(sample.prompt) < 10:
            score -= 0.2
        elif len(sample.prompt) > 50:
            score += 0.1
        
        if len(sample.prompt) > 1000:
            score -= 0.1  # Too long
        
        # Check completion quality
        if len(sample.completion) < 5:
            score -= 0.3
        elif len(sample.completion) > 50:
            score += 0.2
        
        if len(sample.completion) > 2000:
            score -= 0.1  # Too long
        
        # Check for completeness
        if not sample.prompt or not sample.completion:
            score -= 0.5
        
        # Check for common quality indicators
        completion_lower = sample.completion.lower()
        
        # Positive indicators
        positive_indicators = [
            'therefore', 'because', 'however', 'additionally',
            'furthermore', 'specifically', 'for example'
        ]
        score += sum(0.05 for ind in positive_indicators if ind in completion_lower)
        
        # Negative indicators
        negative_indicators = [
            'i cannot', 'i apologize', 'as an ai', 'i don\'t know',
            '[error]', '[incomplete]', '...'
        ]
        score -= sum(0.15 for ind in negative_indicators if ind in completion_lower)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def filter_quality(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Filter samples by quality threshold
        
        Args:
            samples: List of data samples
        
        Returns:
            Filtered samples
        """
        filtered = [
            sample for sample in samples
            if sample.quality_score >= self.quality_threshold
        ]
        
        logger.info(
            f"Quality filtering: {len(samples)} -> {len(filtered)} "
            f"({len(filtered)/len(samples)*100:.1f}% retained)"
        )
        
        return filtered
    
    def _categorize_samples(
        self,
        samples: List[DataSample]
    ) -> List[DataSample]:
        """Categorize samples if not already categorized"""
        for sample in samples:
            if sample.category is None:
                sample.category = self._infer_category(sample)
        
        return samples
    
    def _infer_category(self, sample: DataSample) -> str:
        """Infer category from sample content"""
        prompt_lower = sample.prompt.lower()
        
        # Define category keywords
        categories = {
            'coding': ['code', 'program', 'function', 'bug', 'implement'],
            'analysis': ['analyze', 'evaluate', 'assess', 'compare'],
            'design': ['design', 'architect', 'plan', 'structure'],
            'testing': ['test', 'qa', 'verify', 'validate'],
            'documentation': ['document', 'explain', 'describe', 'write'],
        }
        
        # Score each category
        scores = {}
        for category, keywords in categories.items():
            scores[category] = sum(1 for kw in keywords if kw in prompt_lower)
        
        # Return category with highest score
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return 'general'
    
    def balance_dataset(
        self,
        samples: List[DataSample]
    ) -> List[DataSample]:
        """
        Balance dataset across categories
        
        Args:
            samples: List of data samples
        
        Returns:
            Balanced samples
        """
        # Group by category
        categories: Dict[str, List[DataSample]] = {}
        for sample in samples:
            category = sample.category or 'general'
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        # Find target size (max category size)
        target_size = max(len(samples_list) for samples_list in categories.values())
        
        logger.info(f"Balancing to target size: {target_size} per category")
        
        # Balance each category
        balanced = []
        for category, category_samples in categories.items():
            if len(category_samples) < target_size:
                # Oversample
                oversampled = self._oversample(category_samples, target_size)
                balanced.extend(oversampled)
                logger.info(
                    f"Category '{category}': {len(category_samples)} -> {len(oversampled)}"
                )
            else:
                balanced.extend(category_samples)
        
        # Shuffle
        random.shuffle(balanced)
        
        return balanced
    
    def _oversample(
        self,
        samples: List[DataSample],
        target_size: int
    ) -> List[DataSample]:
        """Oversample minority class"""
        if len(samples) >= target_size:
            return samples
        
        result = samples.copy()
        remaining = target_size - len(samples)
        
        # Randomly duplicate samples
        for _ in range(remaining):
            sample = random.choice(samples)
            # Create copy with new ID
            duplicate = DataSample(
                id=f"{sample.id}_dup_{random.randint(1000, 9999)}",
                prompt=sample.prompt,
                completion=sample.completion,
                category=sample.category,
                quality_score=sample.quality_score,
                metadata=sample.metadata.copy()
            )
            duplicate.metadata['is_duplicate'] = True
            result.append(duplicate)
        
        return result
    
    def augment_data(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Augment data with variations
        
        Args:
            samples: List of data samples
        
        Returns:
            Augmented samples
        """
        if not self.enable_augmentation:
            return samples
        
        augmented = samples.copy()
        
        # For each sample, create variations
        for sample in samples[:len(samples) // self.augmentation_factor]:
            # Create paraphrased versions
            variations = self._create_variations(sample, n=self.augmentation_factor - 1)
            augmented.extend(variations)
        
        logger.info(f"Augmented dataset: {len(samples)} -> {len(augmented)}")
        
        return augmented
    
    def _create_variations(
        self,
        sample: DataSample,
        n: int = 1
    ) -> List[DataSample]:
        """Create variations of a sample"""
        variations = []
        
        for i in range(n):
            # Simple paraphrasing (in real implementation, use NLP model)
            varied_prompt = self._paraphrase(sample.prompt)
            
            variation = DataSample(
                id=f"{sample.id}_var_{i}",
                prompt=varied_prompt,
                completion=sample.completion,
                category=sample.category,
                quality_score=sample.quality_score * 0.9,  # Slightly lower
                metadata=sample.metadata.copy()
            )
            variation.metadata['is_augmented'] = True
            variation.metadata['original_id'] = sample.id
            
            variations.append(variation)
        
        return variations
    
    def _paraphrase(self, text: str) -> str:
        """
        Simple paraphrasing (placeholder implementation)
        
        Args:
            text: Text to paraphrase
        
        Returns:
            Paraphrased text
        """
        # Simple substitutions for demonstration
        substitutions = {
            'write': 'create',
            'make': 'build',
            'fix': 'repair',
            'check': 'verify',
            'test': 'evaluate',
        }
        
        paraphrased = text
        for original, replacement in substitutions.items():
            if original in paraphrased.lower():
                # Simple case-insensitive replacement
                paraphrased = paraphrased.replace(original, replacement)
                break  # Only one substitution
        
        return paraphrased
    
    def _calculate_diversity(self, samples: List[DataSample]) -> float:
        """
        Calculate dataset diversity score
        
        Args:
            samples: List of data samples
        
        Returns:
            Diversity score (0-1)
        """
        if not samples:
            return 0.0
        
        # Calculate category distribution entropy
        categories = [s.category or 'general' for s in samples]
        category_counts = Counter(categories)
        total = len(samples)
        
        # Shannon entropy
        entropy = 0.0
        for count in category_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplified entropy
        
        # Normalize to [0, 1]
        max_entropy = len(category_counts) ** 0.5
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return diversity
    
    def _sample_to_dict(self, sample: DataSample) -> Dict[str, Any]:
        """Convert DataSample to dictionary"""
        return {
            'id': sample.id,
            'prompt': sample.prompt,
            'completion': sample.completion,
            'category': sample.category,
            'quality_score': sample.quality_score,
            'metadata': sample.metadata
        }
    
    def get_statistics(self, samples: List[DataSample]) -> Dict[str, Any]:
        """Get dataset statistics"""
        categories = Counter(s.category or 'general' for s in samples)
        quality_scores = [s.quality_score for s in samples]
        
        return {
            'total_samples': len(samples),
            'categories': dict(categories),
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'min_quality_score': min(quality_scores) if quality_scores else 0,
            'max_quality_score': max(quality_scores) if quality_scores else 0,
            'diversity_score': self._calculate_diversity(samples)
        }

