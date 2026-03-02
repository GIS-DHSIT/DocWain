"""Tests for profile-level domain tagging."""
import pytest
from unittest.mock import MagicMock, patch
from src.profiles.profile_domain_tagger import (
    ProfileDomainResult,
    compute_profile_domain,
    get_profile_domain,
)


class TestProfileDomainResult:
    def test_defaults(self):
        r = ProfileDomainResult()
        assert r.domain == "generic"
        assert r.is_mixed is False
        assert r.distribution == {}
        assert r.confidence == 0.0
        assert r.total_documents == 0


class TestComputeProfileDomain:
    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_all_same_domain(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_config.MongoDB.DOCUMENTS = "documents"
        mock_config.ProfileDomain.MAJORITY_THRESHOLD = 0.80
        mock_coll = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mock_coll.find.return_value = [
            {"document_domain": "hr"},
            {"document_domain": "hr"},
            {"document_domain": "hr"},
        ]
        result = compute_profile_domain("sub1", "prof1")
        assert result.domain == "hr"
        assert result.is_mixed is False
        assert result.confidence == 1.0
        assert result.total_documents == 3

    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_majority_domain(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_config.MongoDB.DOCUMENTS = "documents"
        mock_config.ProfileDomain.MAJORITY_THRESHOLD = 0.80
        mock_coll = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mock_coll.find.return_value = [
            {"document_domain": "hr"},
            {"document_domain": "hr"},
            {"document_domain": "hr"},
            {"document_domain": "hr"},
            {"document_domain": "invoice"},
        ]
        result = compute_profile_domain("sub1", "prof1")
        assert result.domain == "hr"
        assert result.is_mixed is True
        assert result.confidence == 0.8
        assert result.total_documents == 5

    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_mixed_domain_below_threshold(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_config.MongoDB.DOCUMENTS = "documents"
        mock_config.ProfileDomain.MAJORITY_THRESHOLD = 0.80
        mock_coll = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mock_coll.find.return_value = [
            {"document_domain": "hr"},
            {"document_domain": "hr"},
            {"document_domain": "invoice"},
            {"document_domain": "legal"},
            {"document_domain": "medical"},
        ]
        result = compute_profile_domain("sub1", "prof1")
        assert result.domain == "general"
        assert result.is_mixed is True
        # All 5 docs have document_domain so total is 5
        assert result.total_documents == 5

    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_no_documents(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_config.MongoDB.DOCUMENTS = "documents"
        mock_coll = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mock_coll.find.return_value = []
        result = compute_profile_domain("sub1", "prof1")
        assert result.total_documents == 0

    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    def test_no_db(self, mock_get_db):
        mock_get_db.return_value = None
        result = compute_profile_domain("sub1", "prof1")
        assert result.domain == "generic"


class TestGetProfileDomain:
    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_found(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_coll = MagicMock()
        mock_db.get_collection.return_value = mock_coll
        mock_coll.find_one.return_value = {"profile_domain": "invoice"}
        result = get_profile_domain("sub1", "prof1")
        assert result == "invoice"

    @patch("src.profiles.profile_domain_tagger._get_mongo_db")
    @patch("src.api.config.Config")
    def test_not_found(self, mock_config, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_coll = MagicMock()
        mock_db.get_collection.return_value = mock_coll
        mock_coll.find_one.return_value = None
        result = get_profile_domain("sub1", "prof1")
        assert result is None
