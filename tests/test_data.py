"""Tests for data processing utilities."""

from taxembed.data import parse_names_dmp, parse_nodes_dmp


class TestDataParsing:
    """Test data parsing functions."""

    def test_parse_nodes_dmp(self, tmp_path):
        """Test parsing nodes.dmp file."""
        # Create a temporary nodes.dmp file
        nodes_file = tmp_path / "nodes.dmp"
        nodes_file.write_text(
            "1\t|\t1\t|\tno rank\n"  # Root (self-loop, should be skipped)
            "2\t|\t1\t|\tspecies\n"
            "3\t|\t1\t|\tspecies\n"
            "4\t|\t2\t|\tsubspecies\n"
        )

        edges = parse_nodes_dmp(nodes_file)

        # Should have 3 edges (root self-loop excluded)
        assert len(edges) == 3

        # Check structure
        assert all("id1" in e and "id2" in e for e in edges)

    def test_parse_names_dmp(self, tmp_path):
        """Test parsing names.dmp file."""
        # Create a temporary names.dmp file
        names_file = tmp_path / "names.dmp"
        names_file.write_text(
            "1\t|\troot\t|\t\t|\tscientific name\t|\n"
            "2\t|\tBacteria\t|\t\t|\tscientific name\t|\n"
            "3\t|\tArchaea\t|\t\t|\tscientific name\t|\n"
        )

        names_map = parse_names_dmp(names_file)

        assert len(names_map) == 3
        assert names_map[1] == "root"
        assert names_map[2] == "Bacteria"
        assert names_map[3] == "Archaea"
