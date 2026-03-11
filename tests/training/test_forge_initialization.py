"""Tests for Forge class initialization."""

import os

from mlvern.core.forge import Forge
from mlvern.utils.registry import load_registry


class TestForgeInitialization:
    """Tests for Forge class initialization."""

    def test_forge_init_creates_project(self, tmp_mlvern_dir):
        """Test Forge initialization creates project structure."""
        forge = Forge("test_project", tmp_mlvern_dir)
        assert forge.project == "test_project"
        assert forge.base_dir == tmp_mlvern_dir
        assert forge.mlvern_dir == os.path.join(tmp_mlvern_dir, ".mlvern_test_project")

    def test_forge_init_method(self, tmp_mlvern_dir):
        """Test Forge.init() creates required directories."""
        forge = Forge("myproject", tmp_mlvern_dir)
        forge.init()

        # Check directories are created
        assert os.path.exists(os.path.join(forge.mlvern_dir, "datasets"))
        assert os.path.exists(os.path.join(forge.mlvern_dir, "runs"))
        assert os.path.exists(os.path.join(forge.mlvern_dir, "models"))

    def test_forge_init_creates_registry(self, tmp_mlvern_dir):
        """Test Forge.init() creates registry.json."""
        forge = Forge("myproject", tmp_mlvern_dir)
        forge.init()

        registry_path = os.path.join(forge.mlvern_dir, "registry.json")
        assert os.path.exists(registry_path)

        registry = load_registry(forge.mlvern_dir)
        assert registry["project"] == "myproject"
        assert "created_at" in registry
        assert "datasets" in registry
        assert "runs" in registry

    def test_forge_multiple_init_idempotent(self, tmp_mlvern_dir):
        """Test that multiple init() calls don't cause errors."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()
        forge.init()  # Should not raise

        assert os.path.exists(os.path.join(forge.mlvern_dir, "datasets"))
