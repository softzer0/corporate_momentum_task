#!/usr/bin/env python3
"""
Test suite for the Document Storage and Summarization API using pytest
"""

import time
from typing import Generator

import pytest
import requests

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30  # seconds for async operations


@pytest.fixture
def api_client() -> Generator[requests.Session, None, None]:
    """Create a requests session for API testing"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/x-www-form-urlencoded"})
    yield session
    session.close()


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing"""
    return """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also be 
    applied to any machine that exhibits traits associated with a human mind such as learning 
    and problem-solving. The ideal characteristic of artificial intelligence is its ability 
    to rationalize and take actions that have the best chance of achieving a specific goal.
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems the 
    ability to automatically learn and improve from experience without being explicitly 
    programmed. Machine learning focuses on the development of computer programs that can 
    access data and use it to learn for themselves.
    """


@pytest.fixture
def large_text() -> str:
    """Large text for testing file-based storage"""
    base_text = """
    Large Document Test: This is a test to demonstrate the optimized file-based storage system 
    that can handle documents of any size without memory limitations. The new system stores 
    documents directly to files and reads them in chunks when needed, preventing memory overflow 
    issues that would occur with in-memory storage.
    """
    return base_text * 100  # Create a large document


@pytest.fixture
def stored_document(api_client: requests.Session, sample_text: str) -> str:
    """Create a document and return its ID for testing"""
    response = api_client.post(
        f"{BASE_URL}/documents", data=sample_text, headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 201
    result = response.json()
    document_id: str = result["document_id"]
    return document_id


class TestHealthAndRoot:
    """Test health and root endpoints"""

    def test_root_endpoint(self, api_client: requests.Session) -> None:
        """Test the root endpoint"""
        response = api_client.get(f"{BASE_URL}/")
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert isinstance(result["message"], str)

    def test_health_endpoint(self, api_client: requests.Session) -> None:
        """Test the health check endpoint"""
        response = api_client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        result = response.json()

        # Verify required fields
        required_fields = ["status", "documents_count", "summaries_count", "api_available"]
        for field in required_fields:
            assert field in result

        assert result["status"] == "healthy"
        assert isinstance(result["documents_count"], int)
        assert isinstance(result["summaries_count"], int)
        assert isinstance(result["api_available"], bool)


class TestDocumentStorage:
    """Test document storage functionality"""

    def test_store_document_success(self, api_client: requests.Session, sample_text: str) -> None:
        """Test successful document storage"""
        response = api_client.post(
            f"{BASE_URL}/documents", data=sample_text, headers={"Content-Type": "text/plain"}
        )

        assert response.status_code == 201
        result = response.json()
        assert "document_id" in result
        assert isinstance(result["document_id"], str)
        assert len(result["document_id"]) > 0

    def test_store_empty_document(self, api_client: requests.Session) -> None:
        """Test storing an empty document - should succeed with empty content"""
        response = api_client.post(
            f"{BASE_URL}/documents", data="", headers={"Content-Type": "text/plain"}
        )

        # Empty documents should be accepted now
        assert response.status_code == 201
        result = response.json()
        assert "document_id" in result

    def test_store_large_document(self, api_client: requests.Session, large_text: str) -> None:
        """Test storing a large document"""
        response = api_client.post(
            f"{BASE_URL}/documents", data=large_text, headers={"Content-Type": "text/plain"}
        )

        assert response.status_code == 201
        result = response.json()
        assert "document_id" in result

    def test_store_document_form_success(
        self, api_client: requests.Session, sample_text: str
    ) -> None:
        """Test successful document storage via form data"""
        form_data = {"text": sample_text}
        response = api_client.post(
            f"{BASE_URL}/documents/form",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 201
        result = response.json()
        assert "document_id" in result
        assert isinstance(result["document_id"], str)
        assert len(result["document_id"]) > 0

    def test_store_document_form_empty_text(self, api_client: requests.Session) -> None:
        """Test storing empty text via form data - should return validation error"""
        form_data = {"text": ""}
        response = api_client.post(
            f"{BASE_URL}/documents/form",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422  # Validation error for empty text

    def test_store_document_form_large_text(
        self, api_client: requests.Session, large_text: str
    ) -> None:
        """Test storing large text via form data"""
        form_data = {"text": large_text}
        response = api_client.post(
            f"{BASE_URL}/documents/form",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 201
        result = response.json()
        assert "document_id" in result

    def test_store_document_form_missing_text_field(self, api_client: requests.Session) -> None:
        """Test form submission without required text field"""
        response = api_client.post(
            f"{BASE_URL}/documents/form",
            data={},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422  # Validation error for missing required field


class TestDocumentRetrieval:
    """Test document retrieval functionality"""

    def test_retrieve_document_success(
        self, api_client: requests.Session, stored_document: str, sample_text: str
    ) -> None:
        """Test successful document retrieval"""
        response = api_client.get(f"{BASE_URL}/documents/{stored_document}")

        assert response.status_code == 200
        result = response.json()
        assert "document_id" in result
        assert "text" in result
        assert result["document_id"] == stored_document
        assert result["text"].strip() == sample_text.strip()

    def test_retrieve_nonexistent_document(self, api_client: requests.Session) -> None:
        """Test retrieving a non-existent document"""
        response = api_client.get(f"{BASE_URL}/documents/nonexistent123")

        assert response.status_code == 404
        result = response.json()
        assert "detail" in result

    def test_document_auto_streaming_large_file(self, api_client: requests.Session) -> None:
        """Test that large documents are automatically streamed"""
        # Create a document just over 1MB to trigger streaming
        # 1MB = 1,048,576 bytes. Let's create ~1.1MB of text
        chunk = "A" * 1000  # 1KB chunk
        large_text = chunk * 1100  # ~1.1MB of text

        response = api_client.post(
            f"{BASE_URL}/documents", data=large_text, headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 201
        doc_id = response.json()["document_id"]

        # Request the document - should automatically stream because it's > 1MB
        get_response = api_client.get(f"{BASE_URL}/documents/{doc_id}")

        assert get_response.status_code == 200
        assert get_response.headers.get("content-type") == "text/plain; charset=utf-8"
        assert "X-Response-Type" in get_response.headers
        assert get_response.headers["X-Response-Type"] == "streamed"

        # Verify content (just check first and last parts to avoid huge string comparison)
        response_text = get_response.text
        assert response_text.startswith("A" * 100)  # Check first 100 chars
        assert response_text.endswith("A" * 100)  # Check last 100 chars
        assert len(response_text) == len(large_text)  # Check length matches

    def test_document_json_response_small_file(
        self, api_client: requests.Session, sample_text: str
    ) -> None:
        """Test that small documents return JSON response"""
        response = api_client.post(
            f"{BASE_URL}/documents", data=sample_text, headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 201
        doc_id = response.json()["document_id"]

        # Request the document - should return JSON for small files
        get_response = api_client.get(f"{BASE_URL}/documents/{doc_id}")
        assert get_response.status_code == 200
        assert get_response.headers.get("content-type") == "application/json"
        result = get_response.json()
        assert result["document_id"] == doc_id
        assert result["text"].strip() == sample_text.strip()

    def test_document_force_streaming_with_header(
        self, api_client: requests.Session, sample_text: str
    ) -> None:
        """Test that small documents can be forced to stream with Accept header"""
        response = api_client.post(
            f"{BASE_URL}/documents", data=sample_text, headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 201
        doc_id = response.json()["document_id"]

        # Request with Accept: text/plain header to force streaming
        headers = {"Accept": "text/plain"}
        get_response = api_client.get(f"{BASE_URL}/documents/{doc_id}", headers=headers)
        assert get_response.status_code == 200
        assert get_response.headers.get("content-type") == "text/plain; charset=utf-8"
        assert "X-Response-Type" in get_response.headers
        assert get_response.headers["X-Response-Type"] == "streamed"
        assert get_response.text.strip() == sample_text.strip()


class TestDocumentListing:
    """Test document listing functionality"""

    def test_list_documents_empty(self, api_client: requests.Session) -> None:
        """Test listing documents when none exist (or check structure)"""
        response = api_client.get(f"{BASE_URL}/documents")

        assert response.status_code == 200
        result = response.json()
        assert "documents" in result
        assert "total_count" in result
        assert isinstance(result["documents"], list)
        assert isinstance(result["total_count"], int)

    def test_list_documents_with_content(
        self, api_client: requests.Session, stored_document: str
    ) -> None:
        """Test listing documents when at least one exists"""
        response = api_client.get(f"{BASE_URL}/documents")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] >= 1

        # Check that our stored document is in the list
        document_ids = [doc["document_id"] for doc in result["documents"]]
        assert stored_document in document_ids

        # Verify document structure
        for doc in result["documents"]:
            required_fields = [
                "document_id",
                "file_size",
                "status",
                "summary_status",
                "has_summary",
            ]
            for field in required_fields:
                assert field in doc


class TestSummarization:
    """Test document summarization functionality"""

    def test_get_summary_eventually_succeeds(
        self, api_client: requests.Session, stored_document: str
    ) -> None:
        """Test that summary generation eventually succeeds"""
        max_attempts = 15
        wait_time = 2

        for _attempt in range(max_attempts):
            response = api_client.get(f"{BASE_URL}/documents/{stored_document}/summary")

            if response.status_code == 200:
                result = response.json()
                assert "document_id" in result
                assert "summary" in result
                assert result["document_id"] == stored_document
                assert isinstance(result["summary"], str)
                assert len(result["summary"]) > 0
                return  # Success!

            elif response.status_code == 202:
                # Still processing, wait and retry
                result = response.json()
                assert "detail" in result
                time.sleep(wait_time)
                continue

            else:
                pytest.fail(f"Unexpected status code: {response.status_code}")

        pytest.fail(
            f"Summary generation did not complete within {max_attempts * wait_time} seconds"
        )

    def test_get_summary_nonexistent_document(self, api_client: requests.Session) -> None:
        """Test getting summary for non-existent document"""
        response = api_client.get(f"{BASE_URL}/documents/nonexistent123/summary")

        assert response.status_code == 404
        result = response.json()
        assert "detail" in result


class TestIntegration:
    """Integration tests covering full workflows"""

    def test_full_document_workflow(self, api_client: requests.Session, sample_text: str) -> None:
        """Test complete workflow: store -> retrieve -> summarize -> list"""
        # 1. Store document
        store_response = api_client.post(
            f"{BASE_URL}/documents", data=sample_text, headers={"Content-Type": "text/plain"}
        )
        assert store_response.status_code == 201
        doc_id = store_response.json()["document_id"]

        # 2. Retrieve document
        get_response = api_client.get(f"{BASE_URL}/documents/{doc_id}")
        assert get_response.status_code == 200
        assert get_response.json()["text"].strip() == sample_text.strip()

        # 3. Test streaming by forcing with header
        stream_response = api_client.get(
            f"{BASE_URL}/documents/{doc_id}", headers={"Accept": "text/plain"}
        )
        assert stream_response.status_code == 200
        assert stream_response.text.strip() == sample_text.strip()
        assert "X-Response-Type" in stream_response.headers
        assert stream_response.headers["X-Response-Type"] == "streamed"

        # 4. Check it appears in document list
        list_response = api_client.get(f"{BASE_URL}/documents")
        assert list_response.status_code == 200
        doc_ids = [doc["document_id"] for doc in list_response.json()["documents"]]
        assert doc_id in doc_ids

        # 5. Eventually get summary
        max_attempts = 10
        for _ in range(max_attempts):
            summary_response = api_client.get(f"{BASE_URL}/documents/{doc_id}/summary")
            if summary_response.status_code == 200:
                summary_result = summary_response.json()
                assert summary_result["document_id"] == doc_id
                assert len(summary_result["summary"]) > 0
                break
            elif summary_response.status_code == 202:
                time.sleep(2)
                continue
            else:
                pytest.fail(f"Unexpected summary response: {summary_response.status_code}")
        else:
            pytest.fail("Summary generation timed out")


class TestPerformance:
    """Performance-related tests"""

    def test_multiple_concurrent_documents(self, api_client: requests.Session) -> None:
        """Test storing multiple documents in sequence"""
        doc_ids: list[str] = []

        for i in range(5):
            text = f"Test document {i} with some content to store and summarize."
            response = api_client.post(
                f"{BASE_URL}/documents", data=text, headers={"Content-Type": "text/plain"}
            )
            assert response.status_code == 201
            doc_ids.append(response.json()["document_id"])

        # Verify all documents can be retrieved
        for doc_id in doc_ids:
            response = api_client.get(f"{BASE_URL}/documents/{doc_id}")
            assert response.status_code == 200


if __name__ == "__main__":
    """Allow running tests directly"""
    pytest.main([__file__, "-v"])
