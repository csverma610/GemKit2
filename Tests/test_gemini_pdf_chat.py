import unittest
from unittest.mock import patch, MagicMock
import os
import pathlib
from gemini_pdf_chat import GeminiPDFChat

class TestGeminiPDFChat(unittest.TestCase):

    @patch('gemini_pdf_chat.GeminiPDFChat._create_client')
    def setUp(self, mock_create_client):
        # Mock the Gemini client creation
        self.mock_client = MagicMock()
        mock_create_client.return_value = self.mock_client
        
        # Set up the GeminiPDFChat instance
        self.chat = GeminiPDFChat(model_name='gemini-test-model')

    def test_init(self):
        self.assertEqual(self.chat.model_name, 'gemini-test-model')
        self.assertIsNone(self.chat.pdf_file)
        self.assertIsNone(self.chat.pdf_path)
        self.assertEqual(self.chat.chat_history, [])
        self.assertIsNotNone(self.chat.client)

    def test_context_manager_cleanup(self):
        """Test that the context manager calls _delete_pdf on exit."""
        with patch.object(self.chat, '_delete_pdf', wraps=self.chat._delete_pdf) as mock_delete:
            with self.chat as chat_instance:
                # Simulate loading a PDF
                chat_instance.pdf_file = MagicMock()
            # __exit__ should be called here, triggering _delete_pdf
            mock_delete.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_load_pdf_success(self, mock_exists):
        pdf_path = 'test.pdf'
        mock_exists.return_value = True
        
        mock_pdf_file = MagicMock()
        mock_pdf_file.name = 'test.pdf'
        self.mock_client.files.upload.return_value = mock_pdf_file
        
        self.chat.load_pdf(pdf_path)
        
        self.mock_client.files.upload.assert_called_once_with(file=pathlib.Path(pdf_path))
        self.assertEqual(self.chat.pdf_file, mock_pdf_file)
        self.assertEqual(self.chat.pdf_path, pathlib.Path(pdf_path))
        self.assertEqual(self.chat.chat_history, [])

    @patch('pathlib.Path.exists')
    def test_load_pdf_file_not_found(self, mock_exists):
        pdf_path = 'non_existent.pdf'
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.chat.load_pdf(pdf_path)

    def test_delete_pdf(self):
        self.chat.pdf_file = MagicMock()
        self.chat.pdf_file.name = 'test.pdf'
        
        self.chat._delete_pdf()
        
        self.mock_client.files.delete.assert_called_once_with(name='test.pdf')
        self.assertIsNone(self.chat.pdf_file)

    def test_build_contents(self):
        self.chat.pdf_file = 'mock_pdf_file'
        self.chat.chat_history = [{'user': 'Hello', 'assistant': 'Hi there!'}]
        
        contents = self.chat._build_contents('How are you?')
        
        expected_contents = [
            'mock_pdf_file',
            {'role': 'user', 'parts': ['Hello']},
            {'role': 'model', 'parts': ['Hi there!']},
            {'role': 'user', 'parts': ['How are you?']}
        ]
        self.assertEqual(contents, expected_contents)

    def test_generate_text(self):
        self.chat.pdf_file = 'mock_pdf_file'
        mock_response = MagicMock()
        mock_response.text = 'I am doing well, thank you!'
        self.mock_client.models.generate_content.return_value = mock_response
        
        response = self.chat.generate_text('How are you?')
        
        self.assertEqual(response, 'I am doing well, thank you!')
        self.assertEqual(self.chat.chat_history, [{'user': 'How are you?', 'assistant': 'I am doing well, thank you!'}])

if __name__ == '__main__':
    unittest.main()
