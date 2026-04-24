import asyncio
import os
from pathlib import Path
from typing import Any

import yaml

from .tools import ToolResult, ToolError, BASH_TIMEOUT


class Skill:
    """A loaded skill with its metadata and execution logic."""

    def __init__(self, name: str, description: str, config: dict, skill_dir: Path):
        self.name = name
        self.description = description
        self.config = config
        self.skill_dir = skill_dir
        self.enabled = True
        self.error: str | None = None

    @property
    def tool_schema(self) -> dict | None:
        """Generate an Anthropic tool_use schema for this skill."""
        params = self.config.get("parameters", {})
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": params if params else {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    async def execute(self, **kwargs) -> str:
        """Execute the skill with given arguments."""
        # Extract agent workspace if provided (from engine.py)
        agent_workspace = kwargs.pop("_workspace", None) or "."
        run_config = self.config.get("run", {})
        run_type = run_config.get("type", "bash")
        cwd = self.skill_dir

        if run_type == "bash":
            command = run_config.get("command", "")
            # Substitute arguments: {key} -> value
            for k, v in kwargs.items():
                command = command.replace(f"{{{k}}}", str(v))
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=BASH_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Skill timed out after {BASH_TIMEOUT}s"
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output += "\nSTDERR:\n" + stderr_text
            return output

        elif run_type == "python":
            script_file = run_config.get("file", "skill.py")
            script_path = cwd / script_file
            if not script_path.exists():
                return f"Skill script not found: {script_file}"
            # Run in isolated namespace
            namespace = {"args": kwargs, "workspace": str(agent_workspace)}
            try:
                exec(script_path.read_text("utf-8"), namespace)
                result = namespace.get("result", "Skill executed (no result returned)")
            except Exception as e:
                result = f"Skill error: {e}"
            return result

        elif run_type == "http":
            import httpx
            url = run_config.get("url", "")
            method = run_config.get("method", "POST").upper()
            headers = run_config.get("headers", {})
            timeout = run_config.get("timeout", 30)
            client = httpx.AsyncClient(timeout=timeout)
            try:
                if method == "GET":
                    resp = await client.get(url, params=kwargs, headers=headers)
                else:
                    resp = await client.post(url, json=kwargs, headers=headers)
                resp.raise_for_status()
                return resp.text[:50000]
            except Exception as e:
                return f"HTTP skill error: {e}"
            finally:
                await client.aclose()

        else:
            return f"Unknown skill run type: {run_type}"


class SkillRegistry:
    """Discovers, loads, and manages skills."""

    def __init__(self, skill_dirs: list[Path]):
        self.skill_dirs = skill_dirs
        self.skills: dict[str, Skill] = {}
        self._load_errors: list[str] = []

    def discover(self):
        """Scan all skill directories and load skills."""
        self.skills = {}
        self._load_errors = []

        for sd in self.skill_dirs:
            if not sd.exists():
                continue
            for entry in sd.iterdir():
                if not entry.is_dir():
                    continue
                # Look for skill.md (primary) or skill.yaml (legacy)
                md_path = entry / "skill.md"
                yaml_path = entry / "skill.yaml"
                if md_path.exists():
                    try:
                        config = self._parse_skill_md(md_path)
                        if not config or not config.get("name"):
                            self._load_errors.append(f"{entry.name}/skill.md: missing 'name'")
                            continue
                        skill = Skill(
                            name=config["name"],
                            description=config.get("description", ""),
                            config=config,
                            skill_dir=entry,
                        )
                        self.skills[skill.name] = skill
                    except Exception as e:
                        self._load_errors.append(f"{entry.name}: {e}")
                elif yaml_path.exists():
                    try:
                        config = yaml.safe_load(yaml_path.read_text("utf-8"))
                        if not config or not config.get("name"):
                            self._load_errors.append(f"{entry.name}/skill.yaml: missing 'name'")
                            continue
                        skill = Skill(
                            name=config["name"],
                            description=config.get("description", ""),
                            config=config,
                            skill_dir=entry,
                        )
                        self.skills[skill.name] = skill
                    except Exception as e:
                        self._load_errors.append(f"{entry.name}: {e}")

    def get_tool_definitions(self) -> list[dict]:
        """Get tool schemas for all enabled skills."""
        return [s.tool_schema for s in self.skills.values() if s.enabled and s.tool_schema]

    def get_skills_info(self) -> list[dict]:
        """Return serializable info about all skills for the frontend."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "enabled": s.enabled,
                "error": s.error,
            }
            for s in self.skills.values()
        ]

    def toggle_skill(self, name: str, enabled: bool) -> bool:
        if name in self.skills:
            self.skills[name].enabled = enabled
            return True
        return False

    @staticmethod
    def _parse_skill_md(path: Path) -> dict | None:
        """Parse YAML frontmatter from a skill.md file."""
        text = path.read_text("utf-8")
        # Frontmatter is between --- delimiters
        if not text.startswith("---"):
            return None
        end = text.find("---", 3)
        if end == -1:
            return None
        frontmatter = text[3:end].strip()
        if not frontmatter:
            return None
        import yaml
        config = yaml.safe_load(frontmatter)
        # Extract description from body if not in frontmatter
        if isinstance(config, dict) and not config.get("description"):
            body = text[end+3:].strip()
            # Use first line as description fallback
            first_line = body.split("\n")[0].strip()
            if first_line:
                config["description"] = first_line.replace("# ", "").replace("## ", "").strip()
        return config

    async def execute_skill(self, skill_name: str, **kwargs) -> str:
        if skill_name not in self.skills:
            return f"Skill not found: {skill_name}"
        skill = self.skills[skill_name]
        if not skill.enabled:
            return f"Skill disabled: {skill_name}"
        return await skill.execute(**kwargs)
