"""Core module implementation."""

# Note

def _lazy_import():
    global GraspAgent, PromptTemplates
    global ReActEngine, ReActTrace, Tool, ToolRegistry
    global ReflectionModule, FailureType, FailureContext, FailureAnalysis, ReflectionResult
    global RetryStrategy, RetryConfig, AdaptiveRetryPolicy, AdaptiveRetryResult, RetryLevel
    global ExperienceLearner, LearningConfig, Experience, ExperienceType, LearningResult
    global AgenticGraspAgent, AgenticGraspConfig, AgenticGraspResult, AgenticGraspState
    
    try:
        from .grasp_agent import GraspAgent
        from .prompts import PromptTemplates
    except ImportError:
        GraspAgent = None
        PromptTemplates = None
    
    # Note

    from .react_engine import ReActEngine, ReActTrace, Tool, ToolRegistry
    from .reflection import (
        ReflectionModule, 
        FailureType, 
        FailureContext, 
        FailureAnalysis,
        ReflectionResult,
    )
    from .retry_strategy import (
        RetryStrategy, 
        RetryConfig, 
        AdaptiveRetryPolicy,
        AdaptiveRetryResult,
        RetryLevel,
    )
    from .experience_learner import (
        ExperienceLearner, 
        LearningConfig, 
        Experience,
        ExperienceType,
        LearningResult,
    )
    from .agentic_grasp import (
        AgenticGraspAgent,
        AgenticGraspConfig,
        AgenticGraspResult,
        AgenticGraspState,
    )

# Note

from .react_engine import ReActEngine, ReActTrace, Tool, ToolRegistry
from .reflection import (
    ReflectionModule, 
    FailureType, 
    FailureContext, 
    FailureAnalysis,
    ReflectionResult,
)
from .retry_strategy import (
    RetryStrategy, 
    RetryConfig, 
    AdaptiveRetryPolicy,
    AdaptiveRetryResult,
    RetryLevel,
)
from .experience_learner import (
    ExperienceLearner, 
    LearningConfig, 
    Experience,
    ExperienceType,
    LearningResult,
)
from .agentic_grasp import (
    AgenticGraspAgent,
    AgenticGraspConfig,
    AgenticGraspResult,
    AgenticGraspState,
)

# Note

try:
    from .grasp_agent import GraspAgent
    from .prompts import PromptTemplates
except ImportError:
    GraspAgent = None
    PromptTemplates = None

__all__ = [
    # Note

    "GraspAgent",
    "PromptTemplates",
    
    # Note

    "ReActEngine",
    "ReActTrace",
    "Tool",
    "ToolRegistry",
    
    # Note

    "ReflectionModule",
    "FailureType",
    "FailureContext",
    "FailureAnalysis",
    "ReflectionResult",
    
    # Note

    "RetryStrategy",
    "RetryConfig",
    "AdaptiveRetryPolicy",
    "AdaptiveRetryResult",
    "RetryLevel",
    
    # Note

    "ExperienceLearner",
    "LearningConfig",
    "Experience",
    "ExperienceType",
    "LearningResult",
    
    # Note

    "AgenticGraspAgent",
    "AgenticGraspConfig",
    "AgenticGraspResult",
    "AgenticGraspState",
]
