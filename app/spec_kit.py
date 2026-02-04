from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.logger import setup_logger
import re

logger = setup_logger(__name__)


class SpecKitService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=settings.google_api_key, temperature=0.7
        )
        self.specs_dir = Path("specs")
        self.specs_dir.mkdir(exist_ok=True)
        logger.info("Spec Kit service initialized")

    def _slugify(self, text: str) -> str:
        """Convert to slug"""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")

    def specify(self, feature_description: str) -> dict:
        """/specify - Create spec.md"""
        logger.info(f"Creating specification for: {feature_description[:50]}...")

        feature_name = self._slugify(feature_description.split(".")[0][:50])
        feature_dir = self.specs_dir / feature_name
        feature_dir.mkdir(exist_ok=True)

        logger.debug(f"Feature directory: {feature_dir}")

        spec_prompt = f"""Generate a feature specification for: {feature_description}

Include:
- Overview
- User stories
- Functional requirements
- Pydantic models
- API endpoints
- LangGraph workflow
- Success criteria"""

        messages = [
            SystemMessage(content="You are a software architect."),
            HumanMessage(content=spec_prompt),
        ]

        try:
            logger.debug("Calling Gemini API for spec generation")
            response = self.llm.invoke(messages)
            spec_content = response.content

            spec_file = feature_dir / "spec.md"
            spec_file.write_text(spec_content)

            logger.info(f"Specification created: {spec_file}")

            return {
                "feature_name": feature_name,
                "spec_file": str(spec_file),
                "content": spec_content,
            }
        except Exception as e:
            logger.error(f"Error creating specification: {e}")
            raise

    def plan(self, feature_name: str) -> dict:
        """/plan - Create contracts and plan"""
        logger.info(f"Creating implementation plan for: {feature_name}")

        feature_dir = self.specs_dir / feature_name
        spec_file = feature_dir / "spec.md"

        if not spec_file.exists():
            logger.error(f"Spec file not found: {spec_file}")
            raise FileNotFoundError("Run /specify first")

        spec_content = spec_file.read_text()
        logger.debug(f"Read spec file: {len(spec_content)} characters")

        # Create contracts
        contracts_dir = feature_dir / "contracts"
        contracts_dir.mkdir(exist_ok=True)

        contracts_prompt = f"""Based on this spec:

{spec_content}

Generate Pydantic models with type hints, validators, and examples."""

        messages = [
            SystemMessage(content="You are a Pydantic expert."),
            HumanMessage(content=contracts_prompt),
        ]

        try:
            logger.debug("Generating Pydantic contracts")
            response = self.llm.invoke(messages)
            contracts_content = response.content

            if "```python" in contracts_content:
                contracts_content = (
                    contracts_content.split("```python")[1].split("```")[0].strip()
                )

            contracts_file = contracts_dir / "models.py"
            contracts_file.write_text(contracts_content)
            logger.info(f"Contracts created: {contracts_file}")

            # Create plan
            plan_prompt = f"""Create implementation plan for:

{spec_content}

Include: architecture, steps, LangGraph design, services, API, testing."""

            messages = [
                SystemMessage(content="You are a software architect."),
                HumanMessage(content=plan_prompt),
            ]

            logger.debug("Generating implementation plan")
            response = self.llm.invoke(messages)
            plan_content = response.content

            plan_file = feature_dir / "plan.md"
            plan_file.write_text(plan_content)
            logger.info(f"Plan created: {plan_file}")

            return {
                "feature_name": feature_name,
                "contracts_file": str(contracts_file),
                "plan_file": str(plan_file),
            }
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            raise

    def task(self, feature_name: str) -> dict:
        """/task - Break into tasks"""
        logger.info(f"Creating task breakdown for: {feature_name}")

        feature_dir = self.specs_dir / feature_name
        spec_file = feature_dir / "spec.md"
        plan_file = feature_dir / "plan.md"

        if not spec_file.exists():
            logger.error(f"Spec file not found: {spec_file}")
            raise FileNotFoundError("Run /specify first")

        spec_content = spec_file.read_text()
        plan_content = plan_file.read_text() if plan_file.exists() else ""

        task_prompt = f"""Break this into tasks:

SPEC: {spec_content}
PLAN: {plan_content}

Format: GitHub issue style with phases, time estimates, dependencies."""

        messages = [
            SystemMessage(content="You are a project manager."),
            HumanMessage(content=task_prompt),
        ]

        try:
            logger.debug("Generating task breakdown")
            response = self.llm.invoke(messages)
            tasks_content = response.content

            tasks_file = feature_dir / "tasks.md"
            tasks_file.write_text(tasks_content)
            logger.info(f"Tasks created: {tasks_file}")

            return {
                "feature_name": feature_name,
                "tasks_file": str(tasks_file),
                "content": tasks_content,
            }
        except Exception as e:
            logger.error(f"Error creating tasks: {e}")
            raise

    def list_features(self) -> list[dict]:
        """List all features"""
        logger.debug("Listing all features")
        features = []
        for feature_dir in self.specs_dir.iterdir():
            if feature_dir.is_dir() and not feature_dir.name.startswith("."):
                features.append(
                    {
                        "name": feature_dir.name,
                        "has_spec": (feature_dir / "spec.md").exists(),
                        "has_plan": (feature_dir / "plan.md").exists(),
                        "has_tasks": (feature_dir / "tasks.md").exists(),
                    }
                )
        logger.info(f"Found {len(features)} features")
        return features


spec_kit_service = SpecKitService()
