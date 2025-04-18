"""Central storage for all review-related prompts"""

class ReviewPrompts:
    DEFAULT_SYSTEM_PROMPT = """You are an expert code reviewer. Provide detailed, actionable feedback:

1. Technical Accuracy
- Reference specific files, functions, and line numbers
- Explain the impact of each change
- Ground suggestions in the code context

2. Completeness
- Cover all changed files and their dependencies
- Analyze potential impacts thoroughly
- Include test coverage requirements

3. Faithfulness
- Base all suggestions on the actual code
- Avoid assumptions without evidence
- Link suggestions to specific code patterns

Keep suggestions practical and implementation-ready."""

    REVIEW_TEMPLATE = """Analyze this PR with concrete evidence:

CURRENT PR:
{current_pr}

SIMILAR PRS:
{similar_prs}

Provide:
1. Summary - Key changes and their purpose
2. File Changes - Specific files needing updates
3. Conflicts - Files with high change frequency
4. Risks - Potential breaking changes with evidence
5. Testing - Required test coverage with file paths
6. Quality - Code smells with exact locations

Reference specific files, functions, and line numbers."""

    @classmethod
    def get_current_prompt(cls):
        """Get current active prompt"""
        return cls.REVIEW_TEMPLATE, cls.DEFAULT_SYSTEM_PROMPT

    @classmethod
    def update_system_prompt(cls, new_prompt):
        """Update system prompt"""
        cls.DEFAULT_SYSTEM_PROMPT = new_prompt
        # Save to persistent storage
        cls.save_prompts()

    @classmethod
    def save_prompts(cls):
        """Save prompts to file"""
        import json
        import os
        from datetime import datetime

        prompt_history = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": cls.DEFAULT_SYSTEM_PROMPT,
            "review_template": cls.REVIEW_TEMPLATE
        }

        # Create prompts directory if it doesn't exist
        if not os.path.exists("prompts/history"):
            os.makedirs("prompts/history")

        # Save to history file
        with open(f"prompts/history/prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(prompt_history, f, indent=2)