"""Core module for multi-channel cellular automata CUDA implementation."""
from .life_engine import MultiChannelEngine, GameOfLifeEngine
from .cuda_kernels import RuleType, BinaryRule, MultiStateRule, MultiChannelRule

__all__ = ['MultiChannelEngine', 'GameOfLifeEngine', 'RuleType', 
           'BinaryRule', 'MultiStateRule', 'MultiChannelRule']